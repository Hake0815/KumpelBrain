instructions_ultra_ball = [
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": 2, "Max": 2}, "FromPosition": 8},
            },
            {
                "InstructionDataType": 4,
                "Payload": {
                    "Filter": {
                        "Operands": [],
                        "LogicalOperator": 0,
                        "IsLeaf": True,
                        "Condition": {"Field": 2, "Operation": 0, "Value": -1},
                    }
                },
            },
        ],
    },
    {
        "InstructionType": 2,
        "Data": [{"InstructionDataType": 1, "Payload": {"TargetSource": 2}}],
    },
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": 0, "Max": 1}, "FromPosition": 7},
            },
            {
                "InstructionDataType": 4,
                "Payload": {
                    "Filter": {
                        "Operands": [],
                        "LogicalOperator": 0,
                        "IsLeaf": True,
                        "Condition": {"Field": 3, "Operation": 1, "Value": 1},
                    }
                },
            },
        ],
    },
    {"InstructionType": 6, "Data": []},
    {
        "InstructionType": 3,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": -1, "Max": -1}, "FromPosition": 10},
            }
        ],
    },
    {
        "InstructionType": 7,
        "Data": [{"InstructionDataType": 5, "Payload": {"PlayerTarget": 0}}],
    },
    {
        "InstructionType": 2,
        "Data": [{"InstructionDataType": 1, "Payload": {"TargetSource": 1}}],
    },
]
instructions_night_stretcher = [
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": 1, "Max": 1}, "FromPosition": 5},
            },
            {
                "InstructionDataType": 4,
                "Payload": {
                    "Filter": {
                        "Operands": [
                            {
                                "Operands": [],
                                "LogicalOperator": 0,
                                "IsLeaf": True,
                                "Condition": {"Field": 3, "Operation": 1, "Value": 1},
                            },
                            {
                                "Operands": [],
                                "LogicalOperator": 0,
                                "IsLeaf": True,
                                "Condition": {"Field": 4, "Operation": 1, "Value": 8},
                            },
                        ],
                        "LogicalOperator": 2,
                        "IsLeaf": False,
                        "Condition": None,
                    }
                },
            },
        ],
    },
    {"InstructionType": 6, "Data": []},
    {
        "InstructionType": 3,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": -1, "Max": -1}, "FromPosition": 10},
            }
        ],
    },
    {
        "InstructionType": 2,
        "Data": [{"InstructionDataType": 1, "Payload": {"TargetSource": 1}}],
    },
]
petty_grudge_instructions = [
    {
        "InstructionType": 0,
        "Data": [
            {"InstructionDataType": 0, "Payload": {"AttackTarget": 0, "Damage": 10}}
        ],
    }
]
dragon_headbutt_instructions = [
    {
        "InstructionType": 0,
        "Data": [
            {"InstructionDataType": 0, "Payload": {"AttackTarget": 0, "Damage": 70}}
        ],
    }
]
recon_directive_instructions = [
    {
        "InstructionType": 5,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": 2, "Max": 2}, "FromPosition": 7},
            }
        ],
    },
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": 1, "Max": 1}, "FromPosition": 4},
            },
            {
                "InstructionDataType": 4,
                "Payload": {
                    "Filter": {
                        "Operands": [],
                        "LogicalOperator": 0,
                        "IsLeaf": True,
                        "Condition": {"Field": 1, "Operation": 0, "Value": -1},
                    }
                },
            },
        ],
    },
    {
        "InstructionType": 3,
        "Data": [
            {
                "InstructionDataType": 2,
                "Payload": {"Amount": {"Min": -1, "Max": -1}, "FromPosition": 10},
            }
        ],
    },
    {
        "InstructionType": 4,
        "Data": [
            {
                "InstructionDataType": 3,
                "Payload": {"ReturnToDeckType": 0, "FromPosition": 9},
            }
        ],
    },
]

instructions_batch = [
    instructions_ultra_ball,
    instructions_night_stretcher,
    petty_grudge_instructions,
    dragon_headbutt_instructions,
    recon_directive_instructions,
]
