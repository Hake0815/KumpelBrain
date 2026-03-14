import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType


_PROTO_MODULE: ModuleType | None = None


def _load_proto_module() -> ModuleType:
    global _PROTO_MODULE
    if _PROTO_MODULE is not None:
        return _PROTO_MODULE

    try:
        _PROTO_MODULE = importlib.import_module("gamecore_serialization_pb2")
        return _PROTO_MODULE
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    proto_dir = repo_root / "KumpelInterface" / "proto"
    proto_file = proto_dir / "gamecore_serialization.proto"
    out_dir = Path(tempfile.mkdtemp(prefix="kumpel_pb2_"))

    subprocess.run(
        [
            "protoc",
            f"-I{proto_dir}",
            f"--python_out={out_dir}",
            str(proto_file),
        ],
        check=True,
    )

    sys.path.insert(0, str(out_dir))
    _PROTO_MODULE = importlib.import_module("gamecore_serialization_pb2")
    return _PROTO_MODULE


def _build_filter_from_dict(filter_dict: dict):
    pb2 = _load_proto_module()
    node = pb2.ProtoBufFilter()
    node.logical_operator = filter_dict["LogicalOperator"]
    node.is_leaf = filter_dict["IsLeaf"]
    condition = filter_dict.get("Condition")
    if condition is not None:
        node.condition.field = condition["Field"]
        node.condition.operation = condition["Operation"]
        node.condition.value = condition["Value"]
    for operand in filter_dict.get("Operands", []):
        node.operands.add().CopyFrom(_build_filter_from_dict(operand))
    return node


def _build_filter_from_payload(filter_payload):
    if isinstance(filter_payload, dict):
        return _build_filter_from_dict(filter_payload)
    if isinstance(filter_payload, list):
        if not filter_payload:
            raise ValueError("Filter payload list cannot be empty")
        if len(filter_payload) == 1:
            return _build_filter_from_dict(filter_payload[0])
        pb2 = _load_proto_module()
        root = pb2.ProtoBufFilter()
        root.is_leaf = False
        root.logical_operator = 0
        for child in filter_payload:
            root.operands.add().CopyFrom(_build_filter_from_dict(child))
        return root
    raise TypeError("Unsupported filter payload type")


def _fill_instruction_data(message, data_dict: dict) -> None:
    data_type = data_dict["InstructionDataType"]
    payload = _extract_payload(data_dict)
    message.instruction_data_type = data_type

    if data_type == 0:
        message.attack_data.attack_target = payload["AttackTarget"]
        message.attack_data.damage = payload["Damage"]
    elif data_type == 1:
        message.discard_data.target_source = payload["TargetSource"]
    elif data_type == 2:
        message.card_amount_data.amount.min = payload["Amount"]["Min"]
        message.card_amount_data.amount.max = payload["Amount"]["Max"]
        message.card_amount_data.from_position = payload["FromPosition"]
    elif data_type == 3:
        message.return_to_deck_type_data.return_to_deck_type = payload["ReturnToDeckType"]
        message.return_to_deck_type_data.from_position = payload["FromPosition"]
    elif data_type == 4:
        message.filter_data.filter.CopyFrom(_build_filter_from_payload(payload["Filter"]))
    elif data_type == 5:
        message.player_target_data.player_target = payload["PlayerTarget"]
    else:
        raise ValueError(f"Unsupported InstructionDataType: {data_type}")


def serialize_instruction_batches(instructions_batch: list[list[dict]]) -> list[list[bytes]]:
    instructions_batch = normalize_instruction_batches(instructions_batch)
    pb2 = _load_proto_module()
    serialized: list[list[bytes]] = []
    for batch in instructions_batch:
        out_batch: list[bytes] = []
        for instruction_dict in batch:
            instruction = pb2.ProtoBufInstruction()
            instruction.instruction_type = instruction_dict["InstructionType"]
            for data_dict in instruction_dict["Data"]:
                _fill_instruction_data(instruction.data.add(), data_dict)
            out_batch.append(instruction.SerializeToString())
        serialized.append(out_batch)
    return serialized


def serialize_condition_batches(conditions_batch: list[list[dict]]) -> list[list[bytes]]:
    conditions_batch = normalize_condition_batches(conditions_batch)
    pb2 = _load_proto_module()
    serialized: list[list[bytes]] = []
    for batch in conditions_batch:
        out_batch: list[bytes] = []
        for condition_dict in batch:
            condition = pb2.ProtoBufCondition()
            condition.condition_type = condition_dict["ConditionType"]
            for data_dict in condition_dict["Data"]:
                _fill_instruction_data(condition.data.add(), data_dict)
            out_batch.append(condition.SerializeToString())
        serialized.append(out_batch)
    return serialized


def serialize_filter_payload(filter_payload) -> bytes:
    return _build_filter_from_payload(filter_payload).SerializeToString()


def _extract_payload(data_dict: dict) -> dict:
    if "Payload" in data_dict:
        return data_dict["Payload"]

    data_type = data_dict["InstructionDataType"]
    if data_type == 0:
        return data_dict["AttackData"]
    if data_type == 1:
        return data_dict["DiscardData"]
    if data_type == 2:
        return data_dict["CardAmountData"]
    if data_type == 3:
        return data_dict["ReturnToDeckTypeData"]
    if data_type == 4:
        return data_dict["FilterData"]
    if data_type == 5:
        return data_dict["PlayerTargetData"]
    raise ValueError(f"Unsupported InstructionDataType: {data_type}")


def _normalize_instruction_data(data_dict: dict) -> dict:
    if "Payload" in data_dict:
        return data_dict
    payload = _extract_payload(data_dict)
    return {
        "InstructionDataType": data_dict["InstructionDataType"],
        "Payload": payload,
    }


def _normalize_instruction_entries(entries: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for entry in entries:
        if "InstructionType" in entry and "Data" in entry:
            normalized.append(
                {
                    "InstructionType": entry["InstructionType"],
                    "Data": [_normalize_instruction_data(data) for data in entry["Data"]],
                }
            )
        elif "Instructions" in entry:
            for nested in entry["Instructions"]:
                normalized.append(
                    {
                        "InstructionType": nested["InstructionType"],
                        "Data": [
                            _normalize_instruction_data(data) for data in nested["Data"]
                        ],
                    }
                )
        else:
            raise ValueError("Unsupported instruction entry format")
    return normalized


def normalize_instruction_batches(instructions_batch: list[list[dict]]) -> list[list[dict]]:
    return [_normalize_instruction_entries(batch) for batch in instructions_batch]


def normalize_condition_batches(conditions_batch: list[list[dict]]) -> list[list[dict]]:
    normalized: list[list[dict]] = []
    for batch in conditions_batch:
        out_batch: list[dict] = []
        for condition in batch:
            out_batch.append(
                {
                    "ConditionType": condition["ConditionType"],
                    "Data": [
                        _normalize_instruction_data(data)
                        for data in condition.get("Data", [])
                    ],
                }
            )
        normalized.append(out_batch)
    return normalized
