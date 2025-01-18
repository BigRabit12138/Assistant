def dict_has_keys_with_types(
        data: dict,
        expected_fields: list[tuple[str, type]]
) -> bool:
    for field, field_type in expected_fields:
        if field not in data:
            return False

        value = data[field]

        if not isinstance(value, field_type):
            return False

    return True
