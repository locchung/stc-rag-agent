"""Xác thực và giới hạn lượt gọi.

Gọi thẳng hàm dependency thay vì dựng cả app: không cần index, không cần Ollama.
"""
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from seatecco_rag import api, config


def request_gia(ip: str = "1.2.3.4"):
  return SimpleNamespace(client=SimpleNamespace(host=ip))


def test_khoa_rong_thi_cho_qua(monkeypatch):
  # chế độ chạy máy mình: chưa đặt SEATECCO_API_KEY
  monkeypatch.setattr(config, "API_KEY", "")
  assert api.check_api_key(None) is None


def test_co_khoa_ma_khong_gui_thi_401(monkeypatch):
  monkeypatch.setattr(config, "API_KEY", "bi-mat")
  with pytest.raises(HTTPException) as e:
    api.check_api_key(None)
  assert e.value.status_code == 401


def test_gui_sai_khoa_thi_401(monkeypatch):
  monkeypatch.setattr(config, "API_KEY", "bi-mat")
  with pytest.raises(HTTPException) as e:
    api.check_api_key("sai-roi")
  assert e.value.status_code == 401


def test_gui_dung_khoa_thi_qua(monkeypatch):
  monkeypatch.setattr(config, "API_KEY", "bi-mat")
  assert api.check_api_key("bi-mat") is None


def test_vuot_han_muc_thi_429_kem_retry_after(monkeypatch):
  from seatecco_rag.ratelimit import SlidingWindow
  monkeypatch.setattr(api, "_limiter", SlidingWindow(limit=2, window=60))
  monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 2)

  api.check_rate_limit(request_gia(), None)
  api.check_rate_limit(request_gia(), None)
  with pytest.raises(HTTPException) as e:
    api.check_rate_limit(request_gia(), None)
  assert e.value.status_code == 429
  assert "Retry-After" in (e.value.headers or {})


def test_dem_theo_api_key_khi_co_va_theo_ip_khi_khong():
  assert api.client_key(request_gia("9.9.9.9"), None) == "ip:9.9.9.9"
  assert api.client_key(request_gia("9.9.9.9"), "khoa-rat-dai-abcdef").startswith("key:")


def test_hai_ip_khac_nhau_khong_an_han_muc_cua_nhau(monkeypatch):
  from seatecco_rag.ratelimit import SlidingWindow
  monkeypatch.setattr(api, "_limiter", SlidingWindow(limit=1, window=60))
  api.check_rate_limit(request_gia("1.1.1.1"), None)
  api.check_rate_limit(request_gia("2.2.2.2"), None)     # không raise
