## Expected data sample format

```json
{
  "gt_parse": ...
}
```

RVL-CDIP Tiny for Donut:
https://huggingface.co/datasets/nielsr/rvl_cdip_10_examples_per_class_donut

CORD example:
https://huggingface.co/datasets/naver-clova-ix/cord-v2

Other examples:
https://huggingface.co/naver-clova-ix

### Class tokens

JSON values with key `class` will be treated as special class tokens (e.g. `<class_name/>`).

```json
{
  ...
    {
      "class": "class_name",
    },
  ...
}
```
