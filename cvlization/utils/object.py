def getattr_recursively(obj, key, use_first_value_for_repeated_field=True):
    if not key:
        return obj
    x = obj
    for k in key.split("."):
        if use_first_value_for_repeated_field:
            if hasattr(x, "add"):
                if x:
                    x = x[0]
                else:
                    x = x.add()
        x = getattr(x, k)
        # print(f"field: {k}, value: {x}, type: {type(x)}")
    return x


def setattr_recursively(obj, key, value):
    key_path = key.split(".")
    prefix = ".".join(key_path[:-1])
    last_key = key_path[-1]
    x = getattr_recursively(obj, prefix)
    try:
        setattr(x, last_key, value)
    except:
        print("type of x:", type(x))
        print("prefix:", prefix)
        print("key:", last_key)
        print("value of x:", x)
        print("setting to value:", value)
        raise
