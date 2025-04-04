## Non relational database


## Bảng tính
- **Bảng tính** là cách phổ biến để lưu trữ và khám phá dữ liệu.
- **Workbook**: Tập tin bảng tính chứa một hoặc nhiều **Worksheet**.
- **Cell**: Giao điểm giữa hàng và cột, chứa dữ liệu thực tế.
- **Công thức**: Thực hiện các phép toán trên dữ liệu trong cell (bắt đầu bằng dấu "=").
- **Hàm**: Công thức định sẵn, yêu cầu các tham số để thực hiện phép toán (ví dụ: `SUM` để tính tổng).
### 1. Tính toán
- **SUM**: `=SUM(number1, ...)` - Tính tổng.
- **MIN/MAX**: `=MIN(number1, ...)` / `=MAX(number1, ...)` - Giá trị nhỏ nhất/lớn nhất.
- **COUNT/COUNTA**: `=COUNT(value1, ...)` / `=COUNTA(value1, ...)` - Đếm số ô có số/không rỗng.
- **AVERAGE**: `=AVERAGE(number1, ...)` - Tính trung bình.
- **PRODUCT**: `=PRODUCT(number1, ...)` - Tính tích.

### 2. Điều kiện Logic
- **COUNTIF**: `=COUNTIF(range, criteria)` - Đếm ô thỏa điều kiện.
- **IF**: `=IF(logical_test, value_if_true, value_if_false)` - Điều kiện đúng/sai.
- **SUMIF**: `=SUMIF(range, criteria, [sum_range])` - Tính tổng ô thỏa điều kiện.

### 3. Văn bản
- **LEFT/RIGHT/MID**: `=LEFT(text, n)`, `=RIGHT(text, n)`, `=MID(text, start, n)` - Lấy ký tự từ chuỗi.
- **CONCAT**: `=CONCAT(text1, ...)` - Kết hợp chuỗi.

### 4. Ngày tháng
- **NOW**: `=NOW()` - Ngày giờ hiện tại.
- **DATE**: `=DATE(year, month, day)` - Tạo ngày.

### 5. Tra cứu dữ liệu
- **VLOOKUP**: `=VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])` - Tìm kiếm theo cột.
- **INDEX**: `=INDEX(array, row_num, [column_num])` - Trả giá trị từ ô cụ thể.
- **MATCH**: `=MATCH(lookup_value, lookup_array, [match_type])` - Vị trí giá trị trong khoảng.

- **CHOOSE**: `=CHOOSE(index_num, value1, ...)` - Chọn giá trị theo chỉ số.
"""


## NoSQL
- **NoSQL**: Thuật ngữ chung cho các hệ thống lưu trữ dữ liệu không quan hệ.
- Có 4 loại cơ bản:
  1. **Cơ sở dữ liệu key-value**: Lưu trữ cặp khóa-giá trị.
  2. **Cơ sở dữ liệu đồ thị**: Mô tả các mối quan hệ giữa các thực thể (node và edge).
  3. **Cơ sở dữ liệu dạng cột**: Tổ chức dữ liệu theo cột và hàng, chia thành các nhóm gọi là column family.
  4. **Cơ sở dữ liệu tài liệu**: Dựa trên khái niệm key-value, bao gồm các trường và đối tượng.

## Document Data Stores with the Azure Cosmos DB

### 1. Giới thiệu về Azure Cosmos DB
Azure Cosmos DB là một dịch vụ cơ sở dữ liệu đa mô hình được quản lý hoàn toàn, cung cấp khả năng mở rộng và độ sẵn sàng cao.  
Hỗ trợ nhiều mô hình dữ liệu, bao gồm tài liệu, đồ thị, cột và khóa-giá trị.

### 2. Document Data Stores
Document Data Store lưu trữ dữ liệu dưới dạng tài liệu JSON.  
Thích hợp cho các ứng dụng cần lưu trữ dữ liệu phi cấu trúc hoặc bán cấu trúc.

### 3. Tính năng chính
- **Quy mô tự động**: Có thể mở rộng và thu hẹp theo nhu cầu.
- **Độ trễ thấp**: Thời gian phản hồi nhanh, lý tưởng cho các ứng dụng yêu cầu hiệu suất cao.
- **Mô hình dữ liệu linh hoạt**: Có thể lưu trữ các tài liệu khác nhau mà không cần thay đổi cấu trúc.

### 4. API hỗ trợ
Azure Cosmos DB hỗ trợ nhiều API, bao gồm:
- SQL API
- MongoDB API
- Cassandra API
- Gremlin (đồ thị)
- Table API


## chalenge


### Instructions
The [Coca Cola Co spreadsheet](https://github.com/microsoft/Data-Science-For-Beginners/blob/main/2-Working-With-Data/06-non-relational/CocaColaCo.xlsx) is missing some calculations. Your task is to:

- Calculate the Gross profits of FY '15, '16, '17, and '18
Gross Profit = Net Operating revenues - Cost of goods sold
- Calculate the average of all the gross profits. Try to do this with a function.
Average = Sum of gross profits divided by the number of fiscal years (10)
- Documentation on the AVERAGE function
This is an Excel file, but it should be editable in any spreadsheet platform
Data source credit to Yiyi Wang


You can see the sheet from this link:

[My spreadsheet](https://docs.google.com/spreadsheets/d/19ZHsQbTswYS4WUd888m7PAcQVdgZ8EcF/edit?gid=943468144#gid=943468144)