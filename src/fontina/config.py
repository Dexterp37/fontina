import yaml


def load_config(conf_path: str) -> dict:
    with open(conf_path, "r", encoding="utf8") as c:
        try:
            return yaml.safe_load(c)
        except yaml.YAMLError as e:
            # TODO: provide better error handling
            print(e)
            return {}
