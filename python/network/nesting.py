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
