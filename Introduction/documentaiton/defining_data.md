
## Phân Loại Dữ Liệu

1. **Dữ liệu thô(Raw Data)**: 
   - Là dữ liệu ở trạng thái ban đầu, chưa được phân tích hay sắp xếp.
   - Cần được tổ chức để dễ dàng phân tích.

2. **Dữ liệu định lượng(Quantitative Data)**:
   - Quan sát số, có thể phân tích và đo lường.
   - Ví dụ: dân số, chiều cao, thu nhập.

3. **Dữ liệu định tính(Qualitative Data)**:
   - Dữ liệu không thể đo lường khách quan.
   - Ví dụ: bình luận video, nhãn hiệu xe, màu sắc yêu thích.

4. **Dữ liệu có cấu trúc(Structured Data)**:
   - Được sắp xếp theo hàng và cột.
   - Ví dụ: bảng tính, cơ sở dữ liệu quan hệ.

5. **Dữ liệu phi cấu trúc(Unstructured Data)**:
   - Không thể phân loại theo hàng hoặc cột.
   - Ví dụ: tệp văn bản, tin nhắn văn bản, video.

6. **Dữ liệu bán cấu trúc(Semi-structured)**:
   - Kết hợp giữa có cấu trúc và không có cấu trúc.
   - Ví dụ: HTML, tệp CSV, JSON.

## Nguồn Dữ Liệu (Sources of Data)
Nguồn dữ liệu là nơi mà dữ liệu đó được tạo ra hoặc tồn tại
- **Dữ liệu chính(primary data)**: do người dùng tạo ra.
- **Dữ liệu thứ cấp(secondary data)**: từ nguồn đã thu thập để sử dụng chung.
- **Cơ sở dữ liệu**: nơi lưu trữ dữ liệu, thường sử dụng hệ thống quản lý cơ sở dữ liệu.
- **API**: cho phép chia sẻ dữ liệu qua Internet.

# Challenge Classifying Datasets

1. A company has been acquired and now has a parent company. The data scientists have received a spreadsheet of customer phone numbers from the parent company.

2. A smart watch has been collecting heart rate data from its wearer, and the raw data is in JSON format.

3. A workplace survey of employee morale that is stored in a CSV file.

4. Astrophysicists are accessing a database of galaxies that has been collected by a space probe. The data contains the number of planets within in each galaxy.

5. A personal finance app uses APIs to connect to a user's financial accounts in order to calculate their net worth. They can see all of their transactions in a format of rows and columns and looks similar to a spreadsheet.


| **Dataset**                                   | **Structure Type** | **Value Type** | **Source Type**    |
|-----------------------------------------------|---------------------|----------------|---------------------|
| Data from an acquired company                 | Structured           | Qualitative     | Secondary data       |
| Heart rate data from a smartwatch             | Semi-Structured      | Quantitative    | Primary data         |
| Employee morale survey                        | Semi-structured           | Qualitative     | Primary data         |
| Database of galaxies                           | Structured           | Quantitative    | Secondary data       |
| Personal finance application                   | Structured           | Quantitative    | Primary data       |
