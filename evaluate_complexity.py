import torch
import time
import numpy as np
from torch_geometric.data import Data, Batch

# Import model của bạn (Đảm bảo file chứa class nằm cùng thư mục)
# Giả sử file chứa class là model_kan.py
from graph_attention_ffn_kan_multitask import GraphAttentionKAN

def count_parameters(model):
    """Đếm tổng số lượng tham số (Parameters) và kích thước mô hình (MB)"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Ước tính kích thước mô hình (giả sử dùng float32 = 4 bytes)
    model_size_mb = (total_params * 4) / (1024 ** 2)
    
    return total_params, trainable_params, model_size_mb

def create_dummy_ivn_graph(
    num_nodes=64,       # Số lượng node trung bình trong 1 frame mạng CAN
    num_edges=128,      # Số lượng edge trung bình
    node_feat_dim=16,   # ĐIỀN THÔNG SỐ THỰC TẾ CỦA BẠN VÀO ĐÂY
    edge_attr_dim=4,    # ĐIỀN THÔNG SỐ THỰC TẾ CỦA BẠN VÀO ĐÂY
    num_ids=100,        # Khớp với config
    num_relations=4,    # Khớp với config
    device='cuda'
):
    """Tạo một dummy graph sát với thực tế mạng IVN để test"""
    x = torch.randn(num_nodes, node_feat_dim, device=device)
    
    # Random edge_index (2, num_edges)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_attr = torch.randn(num_edges, edge_attr_dim, device=device)
    edge_type = torch.randint(0, num_relations, (num_edges,), device=device)
    
    # ID token cho node embedding
    id_token = torch.randint(0, num_ids, (num_nodes,), device=device)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type, id_token=id_token)
    
    # Phải tạo thành Batch vì trong code forward có gọi data.batch
    batch_data = Batch.from_data_list([data])
    return batch_data

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu test trên thiết bị: {device}")

    # ==========================================
    # 1. KHỞI TẠO MÔ HÌNH (Dựa trên config JSON)
    # ==========================================
    # LƯU Ý: Các thông số dim đầu vào/đầu ra đang đặt giả định, bạn cần thay bằng số thật của dataset
    NODE_FEAT_DIM = 16 
    EDGE_ATTR_DIM = 4  
    NUM_CLASSES = 2       # Phân loại tấn công / bình thường cho Graph
    NUM_NODE_CLASSES = 2  # Phân loại cho Node (Enable multitask)
    NUM_IDS = 100         # Tổng số lượng ID token

    model = GraphAttentionKAN(
        node_feat_dim=NODE_FEAT_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
        num_classes=NUM_CLASSES,
        num_ids=NUM_IDS,
        num_node_classes=NUM_NODE_CLASSES,
        
        # --- Lấy chuẩn từ JSON setup ---
        hidden_dim=128,
        num_layers=3,
        heads=2,
        id_emb_dim=32,
        rel_emb_dim=8,
        num_relations=4,
        dropout=0.2,
        ffn_ratio=2.0,
        
        block_kan_grid_size=3,
        block_kan_spline_order=3,
        block_kan_scale_noise=0.1,
        block_kan_scale_base=1.0,
        block_kan_scale_spline=1.0,
        
        kan_hidden=64,
        kan_grid_size=3,
        kan_spline_order=3,
        kan_scale_noise=0.1,
        kan_scale_base=1.0,
        kan_scale_spline=1.0
    ).to(device)

    model.eval() # Bắt buộc để tắt Dropout

    # Đo số lượng tham số
    total_params, trainable_params, model_size_mb = count_parameters(model)

    # ==========================================
    # 2. CHUẨN BỊ DỮ LIỆU & WARM-UP GPU
    # ==========================================
    # Test với batch_size = 1 để lấy Latency (thời gian phản hồi thực tế của 1 mẫu)
    dummy_input = create_dummy_ivn_graph(
        num_nodes=64, num_edges=200, 
        node_feat_dim=NODE_FEAT_DIM, edge_attr_dim=EDGE_ATTR_DIM, 
        num_ids=NUM_IDS, num_relations=4, device=device
    )

    print("Đang warm-up GPU (tránh hiện tượng chậm ở các vòng đầu)...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input, return_node_logits=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # ==========================================
    # 3. ĐO LƯỜNG LATENCY VÀ THROUGHPUT
    # ==========================================
    num_iterations = 1000
    latencies = []

    print("Bắt đầu đo đạc Latency...")
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            _ = model(dummy_input, return_node_logits=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize() # Bắt buộc phải có lệnh này
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # Đổi ra ms

    # Tính toán các chỉ số thống kê
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p99_latency = np.percentile(latencies, 99)
    throughput = 1000.0 / avg_latency # Mẫu / giây

    # ==========================================
    # 4. IN BÁO CÁO COMPLEXITY
    # ==========================================
    print("\n" + "="*50)
    print(" BÁO CÁO ĐỘ PHỨC TẠP MÔ HÌNH (MODEL COMPLEXITY)")
    print("="*50)
    print(f"Tổng số tham số (Total Params): {total_params:,}")
    print(f"Tham số huấn luyện (Trainable) : {trainable_params:,}")
    print(f"Kích thước lưu trữ ước tính  : {model_size_mb:.2f} MB")
    print("-" * 50)
    print(f"Điều kiện test: Batch Size = 1 (Real-time Edge Inference)")
    print(f"Latency trung bình (Avg)     : {avg_latency:.3f} ms \t(± {std_latency:.3f} ms)")
    print(f"Latency ở 99% (99th Percent): {p99_latency:.3f} ms \t(Chỉ số quan trọng cho real-time)")
    print(f"Thông lượng (Throughput)     : {throughput:.2f} samples/sec")
    print("="*50)

if __name__ == "__main__":
    main()
