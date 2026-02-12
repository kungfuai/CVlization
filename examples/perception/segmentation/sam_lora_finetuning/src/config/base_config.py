"""Bundled Potluck BaseConfig — Pydantic BaseSettings with CLI arg parsing.

Handles:
- Pydantic BaseSettings with ``parse_args()`` classmethod
- Auto-generates argparse from model fields
- Supports Optional[str], bool (store_true/store_false), standard types
- Env file + CLI arg override
"""

from argparse import ArgumentParser
from typing import Optional, Union, get_args, get_origin

from pydantic import Field
from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    """Base configuration with CLI argument parsing."""

    config_file: Optional[str] = Field(
        default=None, description="Path to configuration file."
    )

    @classmethod
    def parse_args(cls):
        parser = ArgumentParser()
        for name, field in cls.model_fields.items():
            field_type = field.annotation
            cli_name = f"--{name.replace('_', '-')}"

            # Unwrap Optional[X] -> X
            if get_origin(field_type) is Union:
                type_args = get_args(field_type)
                non_none_args = tuple(
                    arg for arg in type_args if arg is not type(None)
                )
                if len(non_none_args) == 1:
                    field_type = non_none_args[0]

            assert field_type

            if field_type is bool:
                # store_true when default is False, store_false when True
                action = "store_true" if not field.default else "store_false"
                parser.add_argument(
                    cli_name,
                    dest=name,
                    action=action,
                    default=field.default,
                    help=field.description,
                )
            else:
                parser.add_argument(
                    cli_name,
                    dest=name,
                    type=field_type,
                    default=field.default,
                    help=field.description,
                )

        args = parser.parse_args()

        if not args.config_file:
            config = cls()
        else:
            print(f"Using config: {args.config_file}")
            with open(args.config_file, "rt") as f:
                config = cls.model_validate_json(f.read())

        settings = config.model_dump()
        for key, value in vars(args).items():
            if value != cls.model_fields[key].default:
                settings[key] = value

        return cls(**settings)

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        "env_file": ".env",
    }
