# Take Note:  (Relational Database)  

## Giới thiệu  
- CSDL quan hệ dựa trên **bảng** (table) với **cột** (column) và **dòng** (row).  
- Tăng khả năng mở rộng, truy vấn và phân tích, đồng thời giảm trùng lặp dữ liệu.  

## Khái Niệm Chính  
- **Primary Key (PK)**:  
  - Duy nhất, nhận diện từng dòng trong bảng.  
  - Không thay đổi, thường được tự động sinh ra.  
- **Foreign Key (FK)**:  
  - Liên kết với **PK** ở bảng khác, kết nối dữ liệu giữa các bảng.  

## Phân Tán Dữ Liệu  
- Dữ liệu chia thành nhiều bảng để:  
  - **Giảm trùng lặp**.  
  - **Tăng hiệu suất**.  
- Ví dụ:  
  - Bảng **cities**: Chứa thông tin về thành phố.  
  - Bảng **rainfall**: Chứa lượng mưa, liên kết với **cities** qua **city_id** (FK).  

## Ngôn Ngữ Truy Vấn SQL  

### Câu lệnh SQL cơ bản  
- **SELECT**: Truy xuất dữ liệu.  
- **WHERE**: Lọc dữ liệu theo điều kiện.  
- **JOIN**: Kết hợp dữ liệu từ nhiều bảng dựa vào FK.  
- **GROUP BY**: Nhóm các bản ghi có cùng giá trị trong các cột nhất định.  
- **ORDER BY**: Sắp xếp kết quả truy vấn theo một hoặc nhiều cột.  
- **INSERT**: Chèn dữ liệu mới vào bảng.  
- **UPDATE**: Cập nhật dữ liệu đã tồn tại trong bảng.  
- **DELETE**: Xóa bản ghi khỏi bảng.  
- **LIMIT**: Giới hạn số lượng bản ghi trả về.   

## Các Loại Quan Hệ Giữa Các Bảng  
- **Một - Một (1:1)**: Một dòng trong bảng A liên kết với một dòng trong bảng B.  
- **Một - Nhiều (1:N)**: Một dòng trong bảng A liên kết với nhiều dòng trong bảng B.  
- **Nhiều - Nhiều (N:M)**: Nhiều dòng trong bảng A có thể liên kết với nhiều dòng trong bảng B, thường sử dụng bảng trung gian.  

## Tính Toán và Tối Ưu Hóa  
- **Chỉ Mục (Index)**: Tăng tốc độ truy vấn dữ liệu.  
- **Tối ưu hóa truy vấn**: Sử dụng các phương pháp như phân tích kế hoạch thực thi (execution plan) để cải thiện hiệu suất truy vấn.

## Các Hệ Quản Trị Cơ Sở Dữ Liệu (DBMS) Phổ Biến  
- **MySQL**: Nguồn mở, phổ biến cho ứng dụng web.  
- **PostgreSQL**: Nguồn mở, hỗ trợ nhiều tính năng nâng cao.  
- **Microsoft SQL Server**: DBMS thương mại, phổ biến trong doanh nghiệp.  
- **Oracle Database**: DBMS thương mại, mạnh mẽ cho các ứng dụng lớn.  

## Tính Nhất Quán và Tính Toàn Vẹn Dữ Liệu  
- **ACID Properties**: Đảm bảo tính toàn vẹn và nhất quán của dữ liệu trong giao dịch:  
  - **Atomicity**: Giao dịch phải hoàn thành toàn bộ hoặc không.  
  - **Consistency**: Dữ liệu phải luôn trong trạng thái hợp lệ.  
  - **Isolation**: Giao dịch độc lập với nhau.  
  - **Durability**: Dữ liệu sẽ được lưu giữ ngay cả khi có sự cố.  




## Challenge
A database's schema is its table design and structure. The airports database has two tables: `cities`, which contains a list of cities in the United Kingdom and Ireland, and `airports`, which contains the list of all airports. Because some cities may have multiple airports, two tables were created to store the information.

## Cities Table
| Column Name | Data Type        | Notes                      |
|-------------|------------------|----------------------------|
| id          | integer (PK)     | Primary Key                |
| city        | text             | Name of the city           |
| country     | text             | Name of the country        |

## Airports Table
| Column Name | Data Type        | Notes                      |
|-------------|------------------|----------------------------|
| id          | integer (PK)     | Primary Key                |
| name        | text             | Name of the airport        |
| code        | text             | Airport code               |
| city_id     | integer (FK)     | Foreign Key referencing `id` in Cities |

## Assignment
Create queries to return the following information:

1. **All city names in the Cities table**:
   ```sql
   SELECT city FROM Cities;
   ```

2. **All cities in Ireland in the Cities table**:
    ```sql
    SELECT city FROM Cities WHERE country = 'Ireland';
    ```
3. **All airport names along with their city and country**:

    ```sql
    SELECT Airports.name, Cities.city, Cities.country 
    FROM Airports 
    JOIN Cities ON Airports.city_id = Cities.id;
    ```
4. **All airports in London, United Kingdom**:

    ```sql
    SELECT Airports.name 
    FROM Airports 
    JOIN Cities ON Airports.city_id = Cities.id 
    WHERE Cities.city LIKE 'London' AND Cities.country LIKE 'United Kingdom';
    ```

