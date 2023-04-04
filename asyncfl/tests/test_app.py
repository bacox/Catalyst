import pytest
def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    return a * 1.0 / b

def multiply_by_two(x):
    return multiply(x, 2)


def divide_by_two(x):
    return divide(x, 2)

@pytest.fixture
def numbers():
    a = 10
    b = 20
    return [a,b]


class TestApp:
    def test_addition(self, numbers):
        res = add(*numbers)
        assert res == numbers[0] + numbers[1]

    def test_multiplication(self, numbers):
        res = multiply_by_two(numbers[0])
        assert res == numbers[1]

    def test_division(self, numbers):
        res = divide_by_two(numbers[1])
        assert res == numbers[0]