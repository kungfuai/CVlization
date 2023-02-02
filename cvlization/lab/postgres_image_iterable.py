import logging
from typing import Union
from hashlib import md5
import json

import psycopg2
from PIL import Image

LOGGER = logging.getLogger(__name__)

class PostgresImageIterable:
    def __init__(self, db_conn_info, dataset_name: str, batch_size=8):
        self._db_conn_info = db_conn_info
        self._db_conn = None
        self.dataset_name = dataset_name
        self.batch_size = batch_size

    def __iter__(self):
        self._setup()
        self.is_train_next = True
        return self

    def __next__(self):
        batch = self._query_batch(is_train=self.is_train_next)
        if len(batch) > 0:
            # Alternate
            self.is_train_next = not self.is_train_next
            return batch
        # Other batch type (train/not train) is empty.
        return self._query_batch(is_train=(not self.is_train_next))

    def close_connection(self):
        """
        Shouldn't have to used this often,
        unless the connection explicitly needs to be closed
        for some reason.
        """
        if self._db_conn is not None:
            self._db_conn.close()
            self._db_conn = None

    def check_exists(self):
        self._connect_db()
        try:
            self._run_query(f"select * from {self.dataset_name} limit 1")
        except Exception as ex:
            if str(ex).startswith(f"relation \"{self.dataset_name}\" does not exist"):
                return False
            raise ex
        return True

    def create(self):
        self._connect_db()

        if self.check_exists():
            return

        tbl = self.dataset_name

        creation_queries = [
            f"""
            create table {tbl}(
                id serial primary key,
                created_at timestamptz not null default now(),
                last_used timestamptz not null default now(),
                file bytea not null,
                md5sum text not null unique,
                meta json not null,
                has_label boolean not null,
                is_active boolean not null default true
            )
            """,
            f"""
            create index {tbl}_last_used on {tbl}(last_used)
            """
        ]
        for query in creation_queries:
            self._run_query(query)

    def add_sample(self, pil_image: Image, metadata_dict: dict, has_label: bool):
        self._connect_db()

        img_bytes = pil_image.tobytes()
        md5sum = md5(img_bytes).hexdigest()
        img_binary = psycopg2.Binary(img_bytes)
        img_meta = json.dumps({
            "width": pil_image.width,
            "height": pil_image.height,
            **metadata_dict,
        })
        try:
            self._run_query(
                f"INSERT INTO {self.dataset_name} (file, md5sum, meta, has_label) VALUES (%s, %s, %s, %s)",
                args=(img_binary, md5sum, img_meta, has_label),
            )
        except Exception as ex:
            if not str(ex).startswith("duplicate"):
                raise ex
            else:
                LOGGER.info(f"Duplicate md5sum: {md5sum}")
        return md5sum

    def get_meta_for_md5sum(self, md5sum):
        query = f"""
        select meta
        from {self.dataset_name}
        where md5sum = %s
        and not is_active
        """
        results = self._run_query(query, args=(md5sum,), columns=("meta",))
        if len(results) == 0:
            return None
        return results[0]["meta"]

    def _setup(self):
        self._connect_db()

    def _connect_db(self):
        if self._db_conn is None:
            # Connect to the database
            self._db_conn = psycopg2.connect(**self._db_conn_info)

    def _run_query(self, query: str, args: tuple=(), columns: Union[None, list, tuple]=()):
        """
        Runs a Postgres query, read or write.
        Returns None if no data is returned from the query.
        If data is returned, it's mapped to `columns`.
        """
        conn_ = self._db_conn
        cur = conn_.cursor()
        try:
            cur.execute(query, args)
            if columns is None or len(columns) == 0:
                data = None
            else:
                data = cur.fetchall()
                # For each row, turn into dict
                data = list(map(lambda row: dict(zip(columns, row)), data))
        finally:
            conn_.commit()
            cur.close()
        return data

    def _query_batch(self, is_train: bool) -> Image:
        query = f"""
        update {self.dataset_name}
        set last_used = now()
        where id in (
            select id from {self.dataset_name}
            where has_label = {is_train}
            and is_active
            order by last_used asc
            limit {self.batch_size}
        ) returning
        id, file, meta
        """
        columns = ["id", "file", "meta"]
        data = self._run_query(query, columns=columns)
        return [
            (
                Image.frombytes(
                    "RGB",
                    (row["meta"]["width"], row["meta"]["height"]),
                    row["file"].tobytes(),
                ),
                row["meta"],
                lambda prediction: self._set_prediction(row["id"], prediction) \
                    if not is_train else lambda: 0,
            )
            for row in data
        ]

    def _set_prediction(self, id, prediction):
        prediction_dict = {
            "prediction": prediction,
        }
        query = f"""
        update {self.dataset_name} set
        meta = (meta::jsonb || '{json.dumps(prediction_dict)}')::json,
        is_active = false
        where id = {id}
        """
        self._run_query(query)
