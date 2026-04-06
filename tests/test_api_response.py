"""Unit tests for APIResponse"""

import pytest
from api_response import APIResponse


def test_api_response_success():
    """Test successful API response creation"""
    data = {"key": "value"}
    response = APIResponse.success(data)

    assert response.status == "success"
    assert response.data == data
    assert response.error == ""


def test_api_response_failure():
    """Test failure API response creation"""
    error_msg = "Something went wrong"
    response = APIResponse.failure(error_msg)

    assert response.status == "failed"
    assert response.error == error_msg
    assert response.data is None


def test_api_response_get_item_success():
    """Test getting data from successful response"""
    data = {"total": 10.50, "items": []}
    response = APIResponse.success(data)

    assert response.data["total"] == 10.50
    assert response.data["items"] == []
    assert response.data.get("missing", "default") == "default"


def test_api_response_get_item_failure():
    """Test getting data from failed response"""
    response = APIResponse.failure("Error")

    assert response.data is None
    assert response.data is None


def test_api_response_bool_conversion():
    """Test boolean conversion of APIResponse"""
    success_response = APIResponse.success({"data": "value"})
    failure_response = APIResponse.failure("Error")

    # APIResponse doesn't implement __bool__, so it's always truthy
    assert bool(success_response) is True
    assert bool(failure_response) is True


def test_api_response_status_check():
    """Test status checking methods"""
    success_response = APIResponse.success({"data": "value"})
    failure_response = APIResponse.failure("Error")

    assert success_response.status == "success"
    assert failure_response.status == "failed"
