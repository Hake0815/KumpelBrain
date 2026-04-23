research_instructions = [
    {
        "InstructionType": 2,
        "Data": [
            {
                "InstructionDataType": 1,
                "AttackData": None,
                "DiscardData": {"TargetSource": 0},
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 3,
            }
        ],
    },
    {
        "InstructionType": 3,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 7, "Max": 7}, "FromPosition": 7},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
]

instructions_ultra_ball = [
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 2, "Max": 2}, "FromPosition": 8},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            },
            {
                "InstructionDataType": 4,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": {
                    "Filter": {
                        "Operands": [],
                        "LogicalOperator": 0,
                        "IsLeaf": True,
                        "Condition": {"Field": 2, "Operation": 0, "Value": -1},
                    }
                },
                "PlayerTargetData": None,
                "PayloadCase": 6,
            },
        ],
    },
    {
        "InstructionType": 2,
        "Data": [
            {
                "InstructionDataType": 1,
                "AttackData": None,
                "DiscardData": {"TargetSource": 2},
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 3,
            }
        ],
    },
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 0, "Max": 1}, "FromPosition": 7},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            },
            {
                "InstructionDataType": 4,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": {
                    "Filter": {
                        "Operands": [],
                        "LogicalOperator": 0,
                        "IsLeaf": True,
                        "Condition": {"Field": 3, "Operation": 1, "Value": 1},
                    }
                },
                "PlayerTargetData": None,
                "PayloadCase": 6,
            },
        ],
    },
    {"InstructionType": 6, "Data": []},
    {
        "InstructionType": 3,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {
                    "Amount": {"Min": -1, "Max": -1},
                    "FromPosition": 10,
                },
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
    {
        "InstructionType": 7,
        "Data": [
            {
                "InstructionDataType": 5,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": {"PlayerTarget": 0},
                "PayloadCase": 7,
            }
        ],
    },
    {
        "InstructionType": 2,
        "Data": [
            {
                "InstructionDataType": 1,
                "AttackData": None,
                "DiscardData": {"TargetSource": 1},
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 3,
            }
        ],
    },
]
instructions_night_stretcher = [
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 1, "Max": 1}, "FromPosition": 5},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            },
            {
                "InstructionDataType": 4,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": {
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
                "PlayerTargetData": None,
                "PayloadCase": 6,
            },
        ],
    },
    {"InstructionType": 6, "Data": []},
    {
        "InstructionType": 3,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {
                    "Amount": {"Min": -1, "Max": -1},
                    "FromPosition": 10,
                },
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
    {
        "InstructionType": 2,
        "Data": [
            {
                "InstructionDataType": 1,
                "AttackData": None,
                "DiscardData": {"TargetSource": 1},
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 3,
            }
        ],
    },
]
petty_grudge_instructions = [
    {
        "InstructionType": 0,
        "Data": [
            {
                "InstructionDataType": 0,
                "AttackData": {"AttackTarget": 0, "Damage": 10},
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 2,
            }
        ],
    }
]
dragon_headbutt_instructions = [
    {
        "InstructionType": 0,
        "Data": [
            {
                "InstructionDataType": 0,
                "AttackData": {"AttackTarget": 0, "Damage": 70},
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 2,
            }
        ],
    }
]

recon_directive_instructions = [
    {
        "InstructionType": 5,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 2, "Max": 2}, "FromPosition": 7},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
    {
        "InstructionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 1, "Max": 1}, "FromPosition": 4},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            },
            {
                "InstructionDataType": 4,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": None,
                "FilterData": {
                    "Filter": {
                        "Operands": [],
                        "LogicalOperator": 0,
                        "IsLeaf": True,
                        "Condition": {"Field": 1, "Operation": 0, "Value": -1},
                    }
                },
                "PlayerTargetData": None,
                "PayloadCase": 6,
            },
        ],
    },
    {
        "InstructionType": 3,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {
                    "Amount": {"Min": -1, "Max": -1},
                    "FromPosition": 10,
                },
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
    {
        "InstructionType": 4,
        "Data": [
            {
                "InstructionDataType": 3,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": None,
                "ReturnToDeckTypeData": {"ReturnToDeckType": 0, "FromPosition": 9},
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 5,
            }
        ],
    },
]

instructions_batch = [
    research_instructions,
    instructions_ultra_ball,
    instructions_night_stretcher,
    petty_grudge_instructions,
    dragon_headbutt_instructions,
    recon_directive_instructions,
]

research_conditions = [
    {
        "ConditionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 1, "Max": 60}, "FromPosition": 7},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    }
]

recon_directive_conditions = [
    {
        "ConditionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 1, "Max": 60}, "FromPosition": 7},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
    {"ConditionType": 0, "Data": []},
]

ultra_ball_conditions = [
    {
        "ConditionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 3, "Max": 60}, "FromPosition": 8},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
    {
        "ConditionType": 1,
        "Data": [
            {
                "InstructionDataType": 2,
                "AttackData": None,
                "DiscardData": None,
                "CardAmountData": {"Amount": {"Min": 1, "Max": 60}, "FromPosition": 7},
                "ReturnToDeckTypeData": None,
                "FilterData": None,
                "PlayerTargetData": None,
                "PayloadCase": 4,
            }
        ],
    },
]

conditions_batch = [
    research_conditions,
    recon_directive_conditions,
    ultra_ball_conditions,
]
