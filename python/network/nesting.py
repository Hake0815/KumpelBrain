import torch


def flatten(nested_input, traverse_function):
    flattened_input = []
    groups = []
    operators = {}
    for value, group_index, operator in traverse_function(nested_input):
        flattened_input.append(value)
        groups.append(group_index)
        operators[group_index] = operator
    return flattened_input, groups, operators


def traverse_filter(nested_input, path_list=None):
    if path_list is None:
        path_list = []

    for i, node in enumerate(nested_input):
        path_list.append(str(i))
        if node["IsLeaf"]:
            yield (
                [
                    node["Condition"]["Field"],
                    node["Condition"]["Operation"],
                    node["Condition"]["Value"],
                ],
                tuple(path_list),
                0,
            )
        else:
            yield from traverse_filter_node(
                node["Operands"], path_list, node["LogicalOperator"]
            )
        path_list.pop()  # Backtrack


def traverse_filter_node(nested_input, path_list, operator):
    for i, node in enumerate(nested_input):
        if node["IsLeaf"]:
            yield (
                [
                    node["Condition"]["Field"],
                    node["Condition"]["Operation"],
                    node["Condition"]["Value"],
                ],
                tuple(path_list),
                operator,
            )
        else:
            path_list.append(str(i))
            yield from traverse_filter_node(
                node["Operands"], path_list, node["LogicalOperator"]
            )
            path_list.pop()  # Backtrack


def traverse_filter_v2(nested_input, path_list=None):
    if path_list is None:
        path_list = []

    for i, node in enumerate(nested_input):
        path_list.append(i)
        if node["IsLeaf"]:
            yield (
                [
                    node["Condition"]["Field"],
                    node["Condition"]["Operation"],
                    node["Condition"]["Value"],
                ],
                tuple(path_list),
                0,
            )
        else:
            yield from traverse_filter_node_v2(
                node["Operands"], path_list, node["LogicalOperator"]
            )
        path_list.pop()  # Backtrack


def traverse_filter_node_v2(nested_input, path_list, operator):
    for i, node in enumerate(nested_input):
        if node["IsLeaf"]:
            yield (
                [
                    node["Condition"]["Field"],
                    node["Condition"]["Operation"],
                    node["Condition"]["Value"],
                ],
                tuple(path_list),
                operator,
            )
        else:
            path_list.append(i)
            yield from traverse_filter_node_v2(
                node["Operands"], path_list, node["LogicalOperator"]
            )
            path_list.pop()  # Backtrack


def is_prefix(prefix, test):
    return len(prefix) <= len(test) and test[: len(prefix)] == prefix


def add_to_stack(
    element,
    current_combination,
    current_groups,
    group_index,
    operators: dict,
    combine_function: callable,
):
    if current_groups and group_index == current_groups[-1]:
        current_combination[-1].append(element)
        return

    # Break down stack until we find the right place
    break_down_stack(
        current_combination, current_groups, group_index, operators, combine_function
    )

    # After breaking down, we know where to add
    if current_groups and group_index == current_groups[-1]:
        current_combination[-1].append(element)
    else:
        current_combination.append([element])
        current_groups.append(group_index)


def break_down_stack(
    current_combination,
    current_groups,
    group_index,
    operators: dict,
    combine_function: callable,
):
    while current_groups and not is_prefix(current_groups[-1], group_index):
        current_group = current_groups.pop()
        combined = combine_function(
            current_combination.pop(), operators.get(current_group)
        )
        reduced_index = current_group[:-1]
        add_to_stack(
            combined,
            current_combination,
            current_groups,
            reduced_index,
            operators,
            combine_function,
        )


def reduce(
    flattened_input: torch.Tensor,
    groups: list[tuple[str]],
    operators: dict,
    combine_function: callable,
) -> torch.Tensor:

    current_combination = []
    current_groups = []
    current_batch_index = 0

    for i in range(len(groups)):
        if not current_groups:
            current_groups.append((str(current_batch_index),))
            current_batch_index += 1
            current_combination.append([])

        group_index = groups[i]
        add_to_stack(
            flattened_input[i],
            current_combination,
            current_groups,
            group_index,
            operators,
            combine_function,
        )

    break_down_stack(
        current_combination, current_groups, (), operators, combine_function
    )
    if not current_combination:
        return []
    return current_combination.pop()


def reduce_v2(
    flattened_input: torch.Tensor,
    groups: list[tuple[int]],
    operators: dict,
    combine_function: callable,
) -> torch.Tensor:

    current_combination = []
    current_groups = []
    current_batch_index = 0

    for i in range(len(groups)):
        if not current_groups:
            current_groups.append((current_batch_index,))
            current_batch_index += 1
            current_combination.append([])

        group_index = groups[i]
        add_to_stack(
            flattened_input[i],
            current_combination,
            current_groups,
            group_index,
            operators,
            combine_function,
        )

    break_down_stack(
        current_combination, current_groups, (), operators, combine_function
    )
    if not current_combination:
        return []
    return current_combination.pop()


def vectorize_amount_data(
    amount_data: dict, device: torch.device = None, dtype: torch.dtype = None
) -> torch.Tensor:
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.tensor(
        [
            amount_data["Amount"]["Min"],
            amount_data["Amount"]["Max"],
            amount_data["FromPosition"],
        ],
        **factory_kwargs,
    )


def vectorize_attack_data(
    attack_data: dict, device: torch.device = None, dtype: torch.dtype = None
) -> torch.Tensor:
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.tensor(
        [attack_data["AttackTarget"], attack_data["Damage"]], **factory_kwargs
    )


def vectorize_discard_data(
    discard_data: dict, device: torch.device = None, dtype: torch.dtype = None
) -> torch.Tensor:
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.tensor([discard_data["TargetSource"]], **factory_kwargs)


def vectorize_return_to_deck_type_data(
    return_to_deck_type_data: dict,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.tensor(
        [
            return_to_deck_type_data["ReturnToDeckType"],
            return_to_deck_type_data["FromPosition"],
        ],
        **factory_kwargs,
    )


def vectorize_player_target_data(
    player_target_data: dict, device: torch.device = None, dtype: torch.dtype = None
) -> torch.Tensor:
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.tensor([player_target_data["PlayerTarget"]], **factory_kwargs)


def flatten_instructions(
    instructions: list[list[dict]],
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict], torch.Tensor
]:
    """
    Flatten the instructions
    Args:
        instructions: batch of instructions lists (batch_size, num_instructions*)
    Returns:
        tuple of
            tensor of instruction types dim (total number of instructions)
            tensor of instruction indices (batch_index, instruction_index) dim  (total number of instructions, 2)
            tensor of instruction data type except filter data dim (total number of instruction data)
            tensor of instruction data type indices (batch_index, instruction_index, data_index) dim (total number of instruction data, 3)
            tuple of instruction data tensor lists. One tuple entry for each instruction data type.
            tuple of tensor instruction data indices lists.
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    instruction_types = []
    instruction_indices = []
    instruction_data_types = []
    instruction_data_type_indices = []
    instruction_data = ([], [], [], [], [], [])
    instruction_data_indices = ([], [], [], [], [], [])
    for batch_index, batch_instructions in enumerate(instructions):
        for instruction_index, instruction in enumerate(batch_instructions):
            instruction_types.append(instruction["InstructionType"])
            instruction_indices.append((batch_index, instruction_index))
            for data_index, data in enumerate(instruction["Data"]):
                data_type = data["InstructionDataType"]
                instruction_data_types.append(data_type)
                instruction_data_type_indices.append(
                    (batch_index, instruction_index, data_index)
                )
                if data_type == 4:
                    instruction_data[data_type].append(data["Payload"]["Filter"])
                else:
                    instruction_data[data_type].append(
                        vectorize_payload(data["Payload"], data_type, **factory_kwargs)
                    )
                instruction_data_indices[data_type].append(
                    (batch_index, instruction_index, data_index)
                )

    return (
        torch.tensor(instruction_types, **factory_kwargs),
        torch.tensor(instruction_indices, **factory_kwargs),
        torch.tensor(instruction_data_types, **factory_kwargs),
        torch.tensor(instruction_data_type_indices, **factory_kwargs),
        instruction_data,
        instruction_data_indices,
    )


def vectorize_payload(
    payload: dict,
    data_type: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    factory_kwargs = {"device": device, "dtype": dtype}
    match data_type:
        case 0:
            return vectorize_attack_data(payload, **factory_kwargs)
        case 1:
            return vectorize_discard_data(payload, **factory_kwargs)
        case 2:
            return vectorize_amount_data(payload, **factory_kwargs)
        case 3:
            return vectorize_return_to_deck_type_data(payload, **factory_kwargs)
        case 5:
            return vectorize_player_target_data(payload, **factory_kwargs)
        case _:
            raise ValueError(f"Unknown data type: {data_type}")
