def clean_up_json(json_str: str):
    json_str = (
        json_str.replace("\\n", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace('"[{', "[{")
        .replace("\\", "")
        .strip()
    )

    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):]
    if json_str.endswith("```"):
        json_str = json_str[: len(json_str) - len("```")]

    return json_str
