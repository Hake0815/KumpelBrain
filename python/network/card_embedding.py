from sympy.polys.polyconfig import query
import torch.nn as nn
import torch
from mulit_head_attention import MultiHeadAttention
import nesting


class NormalizedLinear(nn.Module):
    def __init__(
        self, d_in: int, d_out: int, divisor: float = 400.0, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.divisor = divisor
        self.linear = nn.Linear(d_in, d_out, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x / self.divisor)


class SharedEmbeddingHolder(nn.Module):
    def __init__(self, dimension_out: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.card_type_embedding = nn.Embedding(4, dimension_out, **factory_kwargs)
        self.card_subtype_embedding = nn.Embedding(10, dimension_out, **factory_kwargs)
        self.hp_embedding = NormalizedLinear(1, dimension_out, **factory_kwargs)


class FilterConditionEmbedding(nn.Module):
    def __init__(
        self,
        shared_embedding_holder: SharedEmbeddingHolder,
        dimension_out: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.shared_embedding_holder = shared_embedding_holder
        self.filter_field_embedding = nn.Embedding(
            6, dimension_out, padding_idx=0, **factory_kwargs
        )
        self.filter_operation_embedding = nn.Embedding(
            5, dimension_out, padding_idx=0, **factory_kwargs
        )
        self.multi_head_attention = MultiHeadAttention(
            dimension_out,
            dimension_out,
            dimension_out,
            dimension_out // 16,
            2,
            bias=False,
            **factory_kwargs,
        )
        self.dimension_out = dimension_out
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        field_type: torch.Tensor,
        comparison_operator: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            field_type (torch.Tensor): The type of the field shape (N)
            comparison_operator (torch.Tensor): The comparison operator to perform on the field shape (N)
            value (torch.Tensor): The value to compare the field to shape (N)

        Returns:
            embedding (torch.Tensor): The embedding of the filter condition shape (N, dimension_out)
        """
        field_embedding = self.filter_field_embedding(field_type)
        operation_embedding = self.filter_operation_embedding(comparison_operator)

        # Initialize value_embedding with zeros (handles cases 0, 1, 2)
        value_embedding = torch.zeros(
            field_type.shape[0],
            self.dimension_out,
            device=self.device,
            dtype=self.dtype,
        )

        # Create masks and get indices for each field type
        mask_type_3 = field_type == 3
        mask_type_4 = field_type == 4
        mask_type_5 = field_type == 5

        # Handle field_type 3: card_type_embedding
        if mask_type_3.any():
            indices_3 = torch.where(mask_type_3)[0]  # Get original indices
            card_type_values = value[indices_3].long()
            card_type_embeds = self.shared_embedding_holder.card_type_embedding(
                card_type_values
            )
            value_embedding[indices_3] = (
                card_type_embeds  # Assign back to original positions
            )

        # Handle field_type 4: card_subtype_embedding
        if mask_type_4.any():
            indices_4 = torch.where(mask_type_4)[0]  # Get original indices
            card_subtype_values = value[indices_4].long()
            card_subtype_embeds = self.shared_embedding_holder.card_subtype_embedding(
                card_subtype_values
            )
            value_embedding[indices_4] = (
                card_subtype_embeds  # Assign back to original positions
            )

        # Handle field_type 5: hp_embedding
        if mask_type_5.any():
            indices_5 = torch.where(mask_type_5)[0]  # Get original indices
            hp_values = (
                value[indices_5].to(dtype=self.dtype).unsqueeze(1)
            )  # Shape: (N_5, 1)
            hp_embeds = self.shared_embedding_holder.hp_embedding(hp_values)
            value_embedding[indices_5] = hp_embeds  # Assign back to original positions

        query = torch.stack(
            [field_embedding, operation_embedding, value_embedding], dim=1
        )
        updated_query = self.multi_head_attention(query, query, query) + query
        return torch.sum(updated_query, dim=1)

    def forward_v2(
        self,
        field_type: torch.Tensor,
        comparison_operator: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            field_type (torch.Tensor): The type of the field shape (N)
            comparison_operator (torch.Tensor): The comparison operator to perform on the field shape (N)
            value (torch.Tensor): The value to compare the field to shape (N)

        Returns:
            embedding (torch.Tensor): The embedding of the filter condition shape (N, dimension_out)
        """
        field_embedding = self.filter_field_embedding(field_type)
        operation_embedding = self.filter_operation_embedding(comparison_operator)

        # Initialize value_embedding with zeros (handles cases 0, 1, 2)
        value_embedding = torch.zeros(
            field_type.shape[0],
            self.dimension_out,
            device=self.device,
            dtype=self.dtype,
        )

        # Create masks and get indices for each field type
        mask_3 = field_type == 3
        mask_4 = field_type == 4
        mask_5 = field_type == 5

        if mask_3.any():
            idx_3 = mask_3.nonzero(as_tuple=True)[0]
            value_embedding[idx_3] = self.shared_embedding_holder.card_type_embedding(
                value[idx_3].long()
            )
        if mask_4.any():
            idx_4 = mask_4.nonzero(as_tuple=True)[0]
            value_embedding[idx_4] = (
                self.shared_embedding_holder.card_subtype_embedding(value[idx_4].long())
            )
        if mask_5.any():
            idx_5 = mask_5.nonzero(as_tuple=True)[0]
            value_embedding[idx_5] = self.shared_embedding_holder.hp_embedding(
                value[idx_5].to(dtype=self.dtype).unsqueeze(1)
            )

        query = torch.stack(
            [field_embedding, operation_embedding, value_embedding], dim=1
        )
        updated_query = self.multi_head_attention(query, query, query) + query
        return torch.sum(updated_query, dim=1)


class FilterEmbedding(nn.Module):
    def __init__(
        self,
        shared_embedding_holder: SharedEmbeddingHolder,
        dimension_out: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dimension_out = dimension_out
        self.logical_operator_embedding = nn.Embedding(
            3, dimension_out, padding_idx=0, **factory_kwargs
        )
        self.filter_condition_embedding = FilterConditionEmbedding(
            shared_embedding_holder, dimension_out, **factory_kwargs
        )
        self.multi_head_attention = MultiHeadAttention(
            dimension_out,
            dimension_out,
            dimension_out,
            dimension_out // 16,
            2,
            bias=False,
            **factory_kwargs,
        )
        self.device = device
        self.dtype = dtype
        self._operator_tensor_cache = (
            {}
        )  # Cache operator tensors to avoid re-allocation

    def forward(self, filter) -> torch.Tensor:
        if not filter:
            return torch.zeros(self.dimension_out, device=self.device, dtype=self.dtype)

        (field_type, comparison_operator, value), groups_indices, operators = (
            self._to_filter_condition_tensors(filter)
        )
        embedded_conditions = self.filter_condition_embedding.forward(
            field_type, comparison_operator, value
        )
        result = nesting.reduce(
            embedded_conditions, groups_indices, operators, self._combine_condition
        )
        return torch.stack(result, dim=0)

    def _to_filter_condition_tensors(self, filter_condition_batch: list[dict]):
        flattened, group_indices, operators = nesting.flatten(
            filter_condition_batch, nesting.traverse_filter
        )
        tensors = torch.tensor(flattened, dtype=torch.int32, device=self.device)
        return torch.unbind(tensors, dim=1), group_indices, operators

    def _combine_condition(
        self, filter_conditions: list[torch.Tensor], operator: int
    ) -> torch.Tensor:
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        embedded_operator = self.logical_operator_embedding.forward(
            torch.tensor(operator, dtype=torch.int32, device=self.device)
        ).unsqueeze(0)
        filter_conditions_stacked = torch.stack(filter_conditions, dim=0)
        query = torch.cat((filter_conditions_stacked, embedded_operator), 0).unsqueeze(
            0
        )
        updated_query = (
            self.multi_head_attention(query, query, query) + query
        ).squeeze(0)
        return updated_query.sum(dim=0)

    def forward_v2(self, filter) -> torch.Tensor:
        if not filter:
            return torch.zeros(self.dimension_out, device=self.device, dtype=self.dtype)

        (field_type, comparison_operator, value), groups_indices, operators = (
            self._to_filter_condition_tensors_v2(filter)
        )
        embedded_conditions = self.filter_condition_embedding.forward_v2(
            field_type, comparison_operator, value
        )
        result = nesting.reduce_v2(
            embedded_conditions, groups_indices, operators, self._combine_condition_v2
        )
        return torch.stack(result, dim=0)

    def _to_filter_condition_tensors_v2(self, filter_condition_batch: list[dict]):
        flattened, group_indices, operators = nesting.flatten(
            filter_condition_batch, nesting.traverse_filter_v2
        )
        tensors = torch.tensor(flattened, dtype=torch.int32, device=self.device)
        return torch.unbind(tensors, dim=1), group_indices, operators

    def _combine_condition_v2(
        self, filter_conditions: list[torch.Tensor], operator: int
    ) -> torch.Tensor:
        if len(filter_conditions) == 1:
            return filter_conditions[0]

        if operator not in self._operator_tensor_cache:
            self._operator_tensor_cache[operator] = torch.tensor(
                operator, dtype=torch.int32, device=self.device
            )

        embedded_operator = self.logical_operator_embedding(
            self._operator_tensor_cache[operator]
        ).unsqueeze(0)

        filter_conditions_stacked = torch.stack(filter_conditions, dim=0)
        query = torch.cat((filter_conditions_stacked, embedded_operator), 0).unsqueeze(
            0
        )
        updated_query = (
            self.multi_head_attention(query, query, query) + query
        ).squeeze(0)
        return updated_query.sum(dim=0)
