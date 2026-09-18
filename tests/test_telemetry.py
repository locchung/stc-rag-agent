"""Ghi log request. Không gọi model, chỉ kiểm phần ghi và phần đọc metadata."""
import json
from types import SimpleNamespace

from seatecco_rag import telemetry


def tin_nhan(response_metadata=None, usage_metadata=None):
  return SimpleNamespace(response_metadata=response_metadata, usage_metadata=usage_metadata)


def test_ghi_mot_dong_json_doc_lai_duoc(tmp_path):
  p = tmp_path / "requests.jsonl"
  telemetry.log_request({"cau_hoi": "Seatecco có bao nhiêu dự án PCCC?", "tool": ["tra_cuu_du_an"]},
                        path=p)
  dong = json.loads(p.read_text(encoding="utf-8").strip())
  assert dong["tool"] == ["tra_cuu_du_an"]
  assert dong["luc"]                      # thời điểm luôn được thêm vào


def test_append_chu_khong_ghi_de(tmp_path):
  p = tmp_path / "requests.jsonl"
  telemetry.log_request({"cau_hoi": "một"}, path=p)
  telemetry.log_request({"cau_hoi": "hai"}, path=p)
  assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_tu_tao_thu_muc_neu_chua_co(tmp_path):
  p = tmp_path / "chua" / "co" / "requests.jsonl"
  telemetry.log_request({"cau_hoi": "x"}, path=p)
  assert p.exists()


def test_doc_ten_model_tu_moi_kieu_khoa():
  # mỗi provider đặt tên khoá một kiểu, đó là lý do phải dò
  assert telemetry.answering_model([tin_nhan({"model_name": "gemini-3.5-flash-lite"})]) \
      == "gemini-3.5-flash-lite"
  assert telemetry.answering_model([tin_nhan({"model": "qwen3.5:2b"})]) == "qwen3.5:2b"


def test_khong_co_metadata_thi_tra_dau_hoi():
  assert telemetry.answering_model([tin_nhan()]) == "?"
  assert telemetry.answering_model([]) == "?"


def test_lay_model_cua_luot_goi_cuoi():
  # fallback nhảy vào ở lượt sau: phải lấy model đã THẬT SỰ trả lời
  msgs = [tin_nhan({"model_name": "gemini-3.5-flash-lite"}), tin_nhan({"model": "qwen3.5:2b"})]
  assert telemetry.answering_model(msgs) == "qwen3.5:2b"


def test_cong_don_token():
  msgs = [tin_nhan(usage_metadata={"input_tokens": 816, "output_tokens": 24}),
          tin_nhan(usage_metadata={"input_tokens": 4057, "output_tokens": 83}),
          tin_nhan()]
  assert telemetry.token_usage(msgs) == (4873, 107)


def test_luon_in_ra_stdout(tmp_path, capsys):
  """stdout là đường chính: Cloud Run và mọi nền tảng khác tự thu từ đó."""
  telemetry.log_request({"cau_hoi": "x", "tool": ["tra_cuu_du_an"]}, path=tmp_path / "r.jsonl")
  ra = capsys.readouterr().out.strip()
  assert json.loads(ra)["tool"] == ["tra_cuu_du_an"]


def test_tat_ghi_file_van_in_stdout(monkeypatch, capsys):
  from seatecco_rag import config
  monkeypatch.setattr(config, "REQUEST_LOG", None)
  telemetry.log_request({"cau_hoi": "x"})
  assert json.loads(capsys.readouterr().out.strip())["cau_hoi"] == "x"


def test_ghi_file_hong_khong_lam_hong_cau_tra_loi(tmp_path, capsys):
  # đĩa chỉ đọc hoặc hết chỗ là chuyện thường trên container
  cham = tmp_path / "la-file" 
  cham.write_text("không phải thư mục", encoding="utf-8")
  telemetry.log_request({"cau_hoi": "x"}, path=cham / "r.jsonl")   # không được ném lỗi
  assert "cau_hoi" in capsys.readouterr().out
