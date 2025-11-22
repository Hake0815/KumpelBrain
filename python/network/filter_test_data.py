# Test data for FilterJson serialized JSON
# Enum values are integers: FilterLogicalOperator (None=0, And=1, Or=2)
#                          FilterType (None=0, True=1, ExcludeSource=2, CardType=3, CardSubtype=4, Hp=5)
#                          FilterOperation (None=0, Equals=1, NotEquals=2, LessThanOrEqual=3, GreaterThanOrEqual=4)

# Simple leaf node examples
test_data_simple = [
    # Single condition: CardType equals 1
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {"Field": 3, "Operation": 1, "Value": 1},  # CardType  # Equals
    },
    # Single condition: Hp greater than or equal to 50
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {
            "Field": 5,  # Hp
            "Operation": 4,  # GreaterThanOrEqual
            "Value": 50,
        },
    },
    # Single condition: CardSubtype equals 3
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {"Field": 4, "Operation": 1, "Value": 3},  # CardSubtype  # Equals
    },
    # Single condition: True (special case, Value = 0)
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {"Field": 1, "Operation": 0, "Value": 0},  # True  # None
    },
    # Single condition: ExcludeSource (Value = 0)
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {"Field": 2, "Operation": 0, "Value": 0},  # ExcludeSource  # None
    },
]

# Nested structures with logical operators
test_data_nested = [
    # AND operator with two conditions
    {
        "Operands": [
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 3,  # CardType
                    "Operation": 1,  # Equals
                    "Value": 2,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 5,  # Hp
                    "Operation": 4,  # GreaterThanOrEqual
                    "Value": 100,
                },
            },
        ],
        "LogicalOperator": 1,  # And
        "IsLeaf": False,
        "Condition": None,
    },
    # OR operator with two conditions
    {
        "Operands": [
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 0,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 1,
                },
            },
        ],
        "LogicalOperator": 2,  # Or
        "IsLeaf": False,
        "Condition": None,
    },
    # Complex nested: (CardType = 1 AND Hp >= 50) OR CardSubtype = 3
    {
        "Operands": [
            {
                "Operands": [
                    {
                        "Operands": [],
                        "LogicalOperator": 0,  # None
                        "IsLeaf": True,
                        "Condition": {
                            "Field": 3,  # CardType
                            "Operation": 1,  # Equals
                            "Value": 1,
                        },
                    },
                    {
                        "Operands": [],
                        "LogicalOperator": 0,  # None
                        "IsLeaf": True,
                        "Condition": {
                            "Field": 5,  # Hp
                            "Operation": 4,  # GreaterThanOrEqual
                            "Value": 50,
                        },
                    },
                ],
                "LogicalOperator": 1,  # And
                "IsLeaf": False,
                "Condition": None,
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 3,
                },
            },
        ],
        "LogicalOperator": 2,  # Or
        "IsLeaf": False,
        "Condition": None,
    },
    # Deep nesting: ((CardType = 0 OR CardType = 1) AND Hp <= 200) OR CardSubtype = 7
    {
        "Operands": [
            {
                "Operands": [
                    {
                        "Operands": [
                            {
                                "Operands": [],
                                "LogicalOperator": 0,  # None
                                "IsLeaf": True,
                                "Condition": {
                                    "Field": 3,  # CardType
                                    "Operation": 1,  # Equals
                                    "Value": 0,
                                },
                            },
                            {
                                "Operands": [],
                                "LogicalOperator": 0,  # None
                                "IsLeaf": True,
                                "Condition": {
                                    "Field": 3,  # CardType
                                    "Operation": 1,  # Equals
                                    "Value": 1,
                                },
                            },
                        ],
                        "LogicalOperator": 2,  # Or
                        "IsLeaf": False,
                        "Condition": None,
                    },
                    {
                        "Operands": [],
                        "LogicalOperator": 0,  # None
                        "IsLeaf": True,
                        "Condition": {
                            "Field": 5,  # Hp
                            "Operation": 3,  # LessThanOrEqual
                            "Value": 200,
                        },
                    },
                ],
                "LogicalOperator": 1,  # And
                "IsLeaf": False,
                "Condition": None,
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 7,
                },
            },
        ],
        "LogicalOperator": 2,  # Or
        "IsLeaf": False,
        "Condition": None,
    },
]

# Edge cases and boundary values
test_data_edge_cases = [
    # Hp at minimum (0)
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {"Field": 5, "Operation": 1, "Value": 0},  # Hp  # Equals
    },
    # Hp near maximum (399)
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {
            "Field": 5,  # Hp
            "Operation": 3,  # LessThanOrEqual
            "Value": 399,
        },
    },
    # All CardType values
    {
        "Operands": [
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 3,  # CardType
                    "Operation": 1,  # Equals
                    "Value": 0,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 3,  # CardType
                    "Operation": 1,  # Equals
                    "Value": 1,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 3,  # CardType
                    "Operation": 1,  # Equals
                    "Value": 2,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 3,  # CardType
                    "Operation": 1,  # Equals
                    "Value": 3,
                },
            },
        ],
        "LogicalOperator": 2,  # Or
        "IsLeaf": False,
        "Condition": None,
    },
    # All CardSubtype values
    {
        "Operands": [
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 0,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 1,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 2,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 3,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 4,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 6,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 7,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 8,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 4,  # CardSubtype
                    "Operation": 1,  # Equals
                    "Value": 9,
                },
            },
        ],
        "LogicalOperator": 2,  # Or
        "IsLeaf": False,
        "Condition": None,
    },
    # NotEquals operations
    {
        "Operands": [],
        "LogicalOperator": 0,  # None
        "IsLeaf": True,
        "Condition": {"Field": 3, "Operation": 2, "Value": 2},  # CardType  # NotEquals
    },
    # Multiple operations on Hp
    {
        "Operands": [
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 5,  # Hp
                    "Operation": 4,  # GreaterThanOrEqual
                    "Value": 50,
                },
            },
            {
                "Operands": [],
                "LogicalOperator": 0,  # None
                "IsLeaf": True,
                "Condition": {
                    "Field": 5,  # Hp
                    "Operation": 3,  # LessThanOrEqual
                    "Value": 150,
                },
            },
        ],
        "LogicalOperator": 1,  # And
        "IsLeaf": False,
        "Condition": None,
    },
]

# Combine all test data
all_test_data = {
    "simple": test_data_simple,
    "nested": test_data_nested,
    "edge_cases": test_data_edge_cases,
}
