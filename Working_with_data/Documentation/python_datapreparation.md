## Tầm quan trọng của cleaning data
- dễ dàng sử dụng và tái sử dụng (Ease of use and reuse): --> dễ dàng tìm kiếm, sử dụng cũng như chia sẻ với người khác
- làm cho dữ liệu có tính nhất quán: đảm bảo mỗi dataset có cùng độ lệch chuẩn (standardization) từ đó chúng vẫn có giá trị khi hợp nhất
-  Đảm bảo độ chính xác của mô hình

## Mục tiêu và chiến lược
- Exploring a dataset: có thể khám phá tập dữ liệu, đựa ra hình ảnh trực quan hóa, đưa ra những ý tưởng để
giải quyết vấn đề ...
- Formatting: Có nhiều thể loại không nhất quán (inconsistence) tùy thuộc và cách nó được biểu diễn 
, có một số loại phổ biến như whitespace, dates, và data types
- Duplications: dữ liệu trùng lặp thường xuất hiện khi join các dataset và nó tạo ra các kết quả không chính xác,
tuy nhiên trong một số trường hợp nó có thể hưu ích
- Missing Data: Chúng ta có thể điền thêm vào hoặc loại bỏ chúng cùng với các giá trị tương ứng..
 ## Dealing with missing data

- Hầu hết các dữ liệu mà chúng ta sử dụng đều bị missing dữ liệu và có thể ảnh hướng đến phân tích cuối cùng và kết quả thực tế
- pandas xử lý các giá trị thiếu bằng hai cách: 
Cách thứ nhất là dùng special value là Nan
Đối với các giá trị missing không phải là số thì dùng None (object)

- Hàm kiểm tra các giá trị null:
- 
        + isnull()
        + notnull()
    ``` python
    import numpy as np

    example1 = pd.Series([0, np.nan, '', None])
    example1.isnull()
    ```
    kết quả:
    ```
    0    False
    1     True
    2    False
    3     True
    dtype: bool
    ```
- Cách loại bỏ các giá trị null: Hàm dropna()
sẽ loại bỏ các hàng chứa giá trị nan, 
ta cũng có thể thêm các tham số sau để loại bỏ theo ý muốn:
        
        + asix
        + how
        + thresh

- Filling null values:
Chúng ta có thể sử dung fillna để thay thế các giá trị null bằng giá trị mà ta mong muốn.

    ```
    df.fillna(0)
    ```
    sẽ thay thế các giá trị null bằng giá trị 0 
    ```
    df.fillna(method='ffill')
    ```
    hoặc lấy giá trị hợp lệ cuối cùng để điền vào giá trị trống. Nếu thực hiện bằng cách này thì cần có thêm
    tham số axis và có thể có trường hợp chưa tồn tại giá trị hợp lệ trước đó thì vẫn tồn tại null value.

## Removing duplicate data
- Phát hiện các giá trị bị duplicate bằng hàm duplicated()
- Loại bỏ các duplicate: drop_duplicates()  hoặc drop_duplicates(['column'])