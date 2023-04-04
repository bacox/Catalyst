import torch
import pytest
from asyncfl.basgd import krum_aggregation, median_aggregation, trmean_aggregation

class TestAggregation:

    @pytest.fixture
    def t1_fixture(self):
        t1 = torch.Tensor([1, 1, 1])
        return [t1]
    
    @pytest.fixture
    def t2_fixture(self):
        t1 = torch.Tensor([1, 1, 1])
        t2 = torch.Tensor([2, 2, 2])
        return [t1, t2]
    
    @pytest.fixture
    def t3_fixture(self):
        t1 = torch.Tensor([1, 1, 1])
        t2 = torch.Tensor([2, 2, 2])
        t3 = torch.Tensor([3, 3, 3])
        return [t1, t2, t3]
    
    @pytest.fixture
    def t4_fixture(self):
        t1 = torch.Tensor([1, 1, 1])
        t2 = torch.Tensor([2, 2, 2])
        t3 = torch.Tensor([3, 3, 3])
        t4 = torch.Tensor([4, 4, 4])
        return [t1, t2, t3, t4]
    
    @pytest.fixture
    def t5_fixture(self):
        t1 = torch.Tensor([1, 1, 1])
        t2 = torch.Tensor([2, 2, 2])
        t3 = torch.Tensor([3, 3, 3])
        t4 = torch.Tensor([4, 4, 4])
        t5 = torch.Tensor([5, 5, 5])
        t6 = torch.Tensor([6, 6, 6])
        return [t1, t2, t3, t4, t5]
    
    @pytest.fixture
    def t6_fixture(self):
        t1 = torch.Tensor([1, 1, 1])
        t2 = torch.Tensor([2, 2, 2])
        t3 = torch.Tensor([3, 3, 3])
        t4 = torch.Tensor([4, 4, 4])
        t5 = torch.Tensor([5, 5, 5])
        t6 = torch.Tensor([6, 6, 6])
        return [t1, t2, t3, t4, t5, t6]

    def test_median_1(self, t1_fixture):
        t1 = t1_fixture[0]
        out = median_aggregation(t1_fixture)
        assert torch.equal(out, t1)

    def test_median_2(self, t2_fixture):
        out = median_aggregation(t2_fixture)
        t1_t2 = torch.mean(torch.stack(t2_fixture), dim=0)
        assert torch.equal(out, t1_t2)

    def test_median_3(self, t3_fixture):
        _t1, t2, _t3 = t3_fixture
        out = median_aggregation(t3_fixture)
        assert torch.equal(out, t2)

    def test_trmean_6_q1(self, t4_fixture):

        out = trmean_aggregation(t4_fixture, 1)
        t = torch.Tensor([2.5, 2.5, 2.5])
        assert torch.equal(out, t)
    
    def test_trmean_3_q1(self, t3_fixture):

        out = trmean_aggregation(t3_fixture, 1)
        t = t3_fixture[1]
        assert torch.equal(out, t)

    def test_krum_aggregation(self, t2_fixture):

        # out = krum_aggregation(t2_fixture, q=0)
        # @TODO: Implement krum_aggregation
        assert True