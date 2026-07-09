import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự đoán Bệnh Tim", layout="wide")

# --- HÀM TIỆN ÍCH ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('heart_model.pkl')
    except:
        return None

def load_data():
    return pd.read_csv('heart.csv')

def save_model(model):
    joblib.dump(model, 'heart_model.pkl')

# --- MENU ĐIỀU HƯỚNG (SIDEBAR) ---
st.sidebar.title("Hệ thống Dự đoán")
menu = ["Homepage", "Input Form (Dự đoán)", "Prediction Results", "Visualizations","So sánh Thực tế vs Dự đoán", "Huấn luyện lại", "Export Report"]
choice = st.sidebar.radio("Chọn chức năng:", menu)

# --- GIAO DIỆN 1: HOMEPAGE ---
if choice == "Homepage":
    st.title("❤️ Hệ thống Dự đoán Bệnh Tim")
    st.image("E:/Heart_Disease_App/benh_tim.jpeg", width=600)
    
    st.markdown("""
    ### Mục tiêu dự án:
    1. **Hỗ trợ chẩn đoán:** Sử dụng Machine Learning để dự đoán nguy cơ mắc bệnh tim.
    2. **Phân tích dữ liệu:** Cung cấp cái nhìn tổng quan về sức khỏe bệnh nhân.
    3. **Tự động hóa:** Giảm thiểu thời gian phân tích thủ công.
    """)
    
    st.info("👈 Hãy chọn 'Input Form' ở menu bên trái để bắt đầu dự đoán.")

# --- GIAO DIỆN 2: INPUT FORM ---
elif choice == "Input Form (Dự đoán)":
    st.title("📝 Nhập thông tin Bệnh nhân")
    
    # Lưu ý: User yêu cầu "Sinh viên mới", nhưng dữ liệu là bệnh tim nên mình dùng "Bệnh nhân" cho hợp lý
    st.markdown("Vui lòng nhập đầy đủ các chỉ số bên dưới:")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Tuổi (Age)", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Giới tính (Sex)", [0, 1], format_func=lambda x: "Nam" if x==1 else "Nữ")
            cp = st.selectbox("Loại đau ngực (CP)", [0, 1, 2, 3])
            trestbps = st.number_input("Huyết áp (Trestbps)", min_value=50, max_value=250, value=120)
            
        with col2:
            chol = st.number_input("Cholesterol (Chol)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Đường huyết (FBS > 120)", [0, 1])
            restecg = st.selectbox("Điện tâm đồ (RestECG)", [0, 1, 2])
            thalach = st.number_input("Nhịp tim tối đa (Thalach)", min_value=50, max_value=250, value=150)

        with col3:
            exang = st.selectbox("Đau ngực khi tập (Exang)", [0, 1])
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
            slope = st.selectbox("Độ dốc (Slope)", [0, 1, 2])
            ca = st.selectbox("Số mạch chính (CA)", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thal", [0, 1, 2, 3])

        submit_button = st.form_submit_button("🔍 Dự đoán ngay")

    # --- LOGIC VALIDATION & DỰ ĐOÁN ---
    if submit_button:
        # Validation Client-side (đã được xử lý một phần bởi min_value/max_value của Streamlit)
        # Kiểm tra thêm logic tùy chỉnh
        valid = True
        if age < 18:
            st.warning("⚠️ Lưu ý: Dữ liệu huấn luyện chủ yếu cho người trưởng thành.")
        
        if valid:
            # Gom dữ liệu thành DataFrame
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                                      columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
            
            # Lưu vào Session State để chuyển sang trang Kết quả
            st.session_state['input_data'] = input_data
            st.session_state['perform_prediction'] = True
            st.success("Đã nhận dữ liệu! Vui lòng chuyển sang tab 'Prediction Results' hoặc chờ hệ thống chuyển hướng.")
            
            # Tự động hiển thị kết quả nhanh ở dưới (User Friendly)
            model = load_model()
            if model:
                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]
                
                st.divider()
                st.subheader("Kết quả nhanh:")
                if prediction == 1:
                    st.error(f"🚨 DỰ BÁO: CÓ NGUY CƠ BỆNH TIM (Tỷ lệ: {prob*100:.2f}%)")
                else:
                    st.success(f"✅ DỰ BÁO: AN TOÀN (Tỷ lệ nguy cơ: {prob*100:.2f}%)")

# --- GIAO DIỆN 3: PREDICTION RESULTS ---
elif choice == "Prediction Results":
    st.title("📊 Kết quả Dự đoán Chi tiết")
    
    if 'input_data' in st.session_state and st.session_state.get('perform_prediction'):
        df_input = st.session_state['input_data']
        model = load_model()
        
        if model:
            prediction = model.predict(df_input)[0]
            prob = model.predict_proba(df_input)
            
            # Hiển thị thông tin đã nhập
            st.write("Thông tin bệnh nhân:")
            st.dataframe(df_input)
            
            st.write("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Nhãn dự đoán", "Bị bệnh" if prediction == 1 else "Bình thường")
            
            with col2:
                st.metric("Độ tin cậy", f"{prob[0][prediction]*100:.2f}%")
            
            if prediction == 1:
                st.error("⚠️ Hệ thống khuyến nghị bệnh nhân nên đi kiểm tra chuyên sâu.")
            else:
                st.success("🎉 Các chỉ số hiện tại cho thấy sức khỏe tim mạch ổn định.")
        else:
            st.error("Chưa tìm thấy mô hình. Vui lòng huấn luyện mô hình trước.")
    else:
        st.info("Chưa có dữ liệu dự đoán. Vui lòng nhập thông tin tại tab 'Input Form'.")

# --- GIAO DIỆN 4: VISUALIZATIONS ---
elif choice == "Visualizations":
    st.title("📈 Biểu đồ Thống kê & Phân tích")
    df = load_data()
    
    tab1, tab2, tab3 = st.tabs(["Phân phối Target", "Tương quan", "Theo độ tuổi"])
    
    with tab1:
        st.subheader("Tỷ lệ người mắc bệnh vs Không mắc bệnh")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='target', data=df, ax=ax1, palette="pastel")
        ax1.set_xticklabels(['Bình thường', 'Bệnh tim'])
        st.pyplot(fig1)
        
    with tab2:
        st.subheader("Ma trận tương quan (Heatmap)")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Phân bố độ tuổi theo tình trạng bệnh")
        # fig3, ax3 = plt.subplots()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x='age', hue='target', multiple="stack", palette="magma", kde=True, ax=ax3)
        st.pyplot(fig3)

# --- GIAO DIỆN 5: HUẤN LUYỆN LẠI ---
elif choice == "Huấn luyện lại":
    st.title("⚙️ Huấn luyện mô hình (Retrain)")
    st.write("Chức năng này dùng để cập nhật 'trí tuệ' cho AI khi dữ liệu đầu vào thay đổi.")
    st.write("Sử dụng thuật toán: **Decision Tree Classifier**")
    
    # Thông báo tham số để bạn yên tâm là nó không bị lỗi 100% nữa
    st.info("Cấu hình: Max Depth = 4, Min Samples Leaf = 30 (Giúp mô hình thực tế hơn, tránh học vẹt).")

    if st.button("🚀 Bắt đầu Huấn luyện"):
        with st.spinner("Đang xử lý dữ liệu và huấn luyện..."):
            # 1. Tải dữ liệu
            df = load_data()
            if df.empty:
                st.error("Không có dữ liệu để huấn luyện!")
                st.stop()

            # 2. Chuẩn bị dữ liệu (X, y)
            X = df.drop('target', axis=1)
            y = df['target']
            
            # 3. Chia tập Train/Test (Bắt buộc random_state=42 để đồng bộ với biểu đồ)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 4. Khởi tạo Model (Đã cài tham số chặn lỗi 100%)
            model = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=30, random_state=42)
            
            # 5. Huấn luyện (Fit) và Lưu (Save)
            model.fit(X_train, y_train)
            save_model(model)
            
            # 6. Đánh giá kết quả
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # 7. Hiển thị kết quả
            st.success(f"Huấn luyện hoàn tất! Độ chính xác thực tế: {acc*100:.2f}%")
            st.write("Báo cáo chi tiết (Classification Report):")
            st.text(classification_report(y_test, y_pred))

# --- GIAO DIỆN 6: XUẤT BÁO CÁO ---
elif choice == "Export Report":
    st.title("📂 Xuất Báo cáo")
    
    df = load_data()
    
    st.subheader("1. Xem trước dữ liệu")
    st.dataframe(df.head(10))
    
    st.subheader("2. Tải xuống dữ liệu")
    
    # Chuyển đổi dataframe thành CSV
    csv = df.to_csv(index=False).encode('utf-8')
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Tải xuống dữ liệu gốc (CSV)",
            data=csv,
            file_name='heart_data_report.csv',
            mime='text/csv',
        )
    
    with col2:
        # Giả lập báo cáo thống kê
        report_text = f"""
        BAO CAO TONG HOP DU AN DU DOAN BENH TIM
        ---------------------------------------
        Tong so mau: {len(df)}
        So ca mac benh: {len(df[df['target']==1])}
        So ca binh thuong: {len(df[df['target']==0])}
        Do tuoi trung binh: {df['age'].mean():.1f}
        """
        st.download_button(
            label="📥 Tải xuống báo cáo tóm tắt (TXT)",
            data=report_text,
            file_name='summary_report.txt',
            mime='text/plain'
        )

# --- GIAO DIỆN MỚI: SO SÁNH THỰC TẾ VS DỰ ĐOÁN ---
elif choice == "So sánh Thực tế vs Dự đoán":
    st.title("🧐 So sánh Kết quả Thực tế và Dự đoán")
    st.write("Bảng này sử dụng dữ liệu gốc để kiểm tra lại độ chính xác của mô hình trên tập kiểm tra (Test Set).")
    
    # 1. Tải model và dữ liệu
    model = load_model()
    df = load_data()
    
    if model is None:
        st.error("Chưa có mô hình! Vui lòng sang tab 'Huấn luyện lại' để tạo mô hình trước.")
    else:
        # 2. Tái tạo lại tập Test giống như lúc huấn luyện (random_state=42)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Lưu ý: Phải dùng random_state=42 y hệt như lúc train để dữ liệu khớp nhau
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Thực hiện dự đoán lại trên tập Test
        y_pred = model.predict(X_test)
        
        # 4. Tạo DataFrame so sánh
        comparison_df = X_test.copy()
        comparison_df['Thực tế'] = y_test
        comparison_df['Dự đoán'] = y_pred
        
        # Tạo cột trạng thái để dễ nhìn
        comparison_df['Kết quả'] = comparison_df.apply(
            lambda row: '✅ Đúng' if row['Thực tế'] == row['Dự đoán'] else '❌ Sai', axis=1
        )
        
        # 5. Hiển thị các chỉ số thống kê
        correct_count = len(comparison_df[comparison_df['Kết quả'] == '✅ Đúng'])
        wrong_count = len(comparison_df[comparison_df['Kết quả'] == '❌ Sai'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng mẫu kiểm tra", len(X_test))
        col2.metric("Số mẫu ĐÚNG", correct_count, delta="Tốt")
        col3.metric("Số mẫu SAI", wrong_count, delta_color="inverse")
        
        st.divider()
        
        # 6. Bộ lọc xem dữ liệu (Chức năng nâng cao)
        filter_option = st.radio("Bạn muốn xem dữ liệu nào?", ["Xem tất cả", "Chỉ xem các ca dự đoán SAI"])
        
        if filter_option == "Chỉ xem các ca dự đoán SAI":
            final_df = comparison_df[comparison_df['Kết quả'] == '❌ Sai']
            st.warning("Dưới đây là danh sách các bệnh nhân mà AI đã dự đoán sai:")
        else:
            final_df = comparison_df
            st.info("Dưới đây là toàn bộ danh sách kiểm tra:")
            
        # Hiển thị bảng
        st.dataframe(final_df, use_container_width=True)
        
        # 7. Vẽ biểu đồ so sánh nhỏ
        st.subheader("Biểu đồ tương quan lỗi")
        fig_comp, ax_comp = plt.subplots()
        # Đếm số lượng đúng sai
        status_counts = comparison_df['Kết quả'].value_counts()
        ax_comp.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
        ax_comp.axis('equal') 
        st.pyplot(fig_comp)