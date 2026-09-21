"""Prompt hệ thống. Để riêng một file vì eval và service phải dùng CHÍNH nó.

Từng có lần prompt trong eval lệch với prompt của agent, và mọi số đo sau đó
đều vô nghĩa. Một nguồn duy nhất thì không lệch được.
"""
from .text import chua

NL = chr(10)

# Câu trả lời khi không có chunk nào đủ liên quan (xem config.SEARCH_MIN_SCORE).
# Ở đây chứ không ở tools.py vì cả ba nơi phải dùng ĐÚNG một chuỗi: tool trả về
# nó, prompt dạy model nhắc lại nguyên văn, evals đếm số lần model chịu nói.
KHONG_TIM_THAY = "Không tìm thấy thông tin này trong tài liệu Seatecco."

# Mọi cách hệ thống nói "tôi không biết" - KHÔNG chỉ câu trên. Ba tool danh sách
# dùng return_direct nên CHÍNH chúng dựng câu từ chối riêng ("Không có dự án nào ở
# 'Nhật Bản' trong dữ liệu."), và đó là câu người dùng đọc.
# evals/run_calibration.py đếm số lần model chịu nói không biết bằng danh sách này,
# nên thêm một cách từ chối mới trong tools.py là phải khai ở đây, kẻo bộ đo chấm
# nó thành "bịa". tests/test_tools.py canh giúp việc đó.
CACH_NOI_KHONG_BIET = [
  "không tìm thấy",
  "không có dự án nào",           # cả bản lọc địa điểm và bản lọc lĩnh vực/hạng mục
  "không có bài tin tức nào",
  "không có tin tuyển dụng nào",
  "không có thông tin",           # ba câu dưới là cách model tự diễn đạt
  "không đề cập",
  "không nêu",
]


def la_noi_khong_biet(answer: str) -> bool:
  """Câu trả lời có phải là "tôi không biết" hay không. Bỏ dấu khi so, nhờ text.chua."""
  return any(chua(cach, answer) for cach in CACH_NOI_KHONG_BIET)

# Ví dụ ngắn (few-shot) đặt ngay trong system prompt: rẻ hơn fine-tune và là cách
# hiệu quả nhất để model nhỏ chọn đúng tool, nhất là câu bẫy và câu ngoài phạm vi.
VI_DU = [
  ("Seatecco có bao nhiêu dự án PCCC?", "gọi tra_cuu_du_an(hang_muc='PCCC')"),
  ("dự án The Poet Residence là gì?", "gọi search_documentation"),
  ("lương công nhân cơ điện lạnh bao nhiêu?", "gọi liet_ke_tuyen_dung"),
  ("xin chào", "chào lại, KHÔNG gọi tool"),
  ("giá vàng hôm nay bao nhiêu?", "gọi search_documentation, tool không có thì "
                                  f"trả lời đúng: {KHONG_TIM_THAY}"),
]

SYSTEM_PROMPT = (
  "Bạn là trợ lý của công ty Seatecco. "
  "Với mọi câu hỏi về Seatecco, LUÔN gọi một tool để tra tài liệu TRƯỚC khi trả lời. "
  "Chỉ trả lời dựa trên kết quả mà tool trả về, bằng tiếng Việt. "
  "Không thêm thông tin nào ngoài kết quả tool. "
  "Chọn tool như sau: câu hỏi liệt kê, đếm dự án, hoặc hỏi công ty CÓ hay KHÔNG có dự án "
  "thuộc một lĩnh vực hay hạng mục nào đó thì gọi tra_cuu_du_an; "
  "hỏi danh sách tin tức thì gọi liet_ke_tin_tuc; "
  "hỏi về tuyển dụng, vị trí đang tuyển, mô tả công việc, lương, quyền lợi, cách nộp hồ sơ "
  "thì gọi liet_ke_tuyen_dung; "
  "các câu hỏi khác, kể cả hỏi chi tiết một dự án cụ thể, gọi search_documentation."
  # --- biết mình không biết: thà nói không tìm thấy còn hơn đoán ---
  + NL + NL
  + "NÓI KHÔNG BIẾT là câu trả lời ĐÚNG, không phải thất bại. "
  + f"Nếu tool trả về đúng câu \"{KHONG_TIM_THAY}\" thì trả lời lại ĐÚNG câu đó, "
  + "không thêm chữ nào, không suy đoán, không dùng kiến thức ngoài tài liệu. "
  + "Nếu kết quả tool có nội dung nhưng KHÔNG chứa dữ kiện được hỏi thì cũng trả lời "
  + f"\"{KHONG_TIM_THAY}\" thay vì lấy một con số gần giống trong đó. "
  + "Thà nói không tìm thấy còn hơn trả lời sai."
  + NL + NL
  + "Ví dụ:" + NL
  + NL.join(f"- \"{hoi}\" -> {lam}" for hoi, lam in VI_DU)
)
