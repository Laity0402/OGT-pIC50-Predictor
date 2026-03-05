# -*- coding: utf-8 -*-
import io
import uuid
import datetime
import traceback
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, DataStructs
from DeepPurpose import utils, CompoundPred

# ========================== 路径配置 ==========================
CURRENT_FILE_PATH = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE_PATH.parent
MODEL_ROOT = BACKEND_DIR.parent / "model"  

if not MODEL_ROOT.exists():
    raise FileNotFoundError(f"The model root directory does not exist, please check the path:{MODEL_ROOT}")
print(f"The model root directory has been confirmed:{MODEL_ROOT}")

# ========================== 模型编码方式映射 ==========================
MODEL_DEFAULT_ENCODING = {
    "rdkit_2d_normalizedModel": "rdkit_2d_normalized",
    "DaylightModel": "Daylight",
    "ErGModel": "ErG",
    "MorganModel": "Morgan"
}

# ========================== Pydantic 模型定义 ==========================
class ResultItem(BaseModel):
    id: str
    smiles: str
    pic50: float
    model_used: str
    timestamp: str
    molecule_name: str
    mol_wt: float
    logp: float
    hbd: int
    hba: int


class PaginatedResults(BaseModel):
    total_items: int
    total_pages: int
    current_page: int
    items: List[ResultItem]
    model_used: Optional[str]  # 返回当前筛选的模型名称

# ========================== 多模型管理 ==========================
class ModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, Any] = {}
        self.default_model: str = None

    def load_model(self, model_name: str, config_filename: str, model_filename: str) -> None:
        try:
            print(f"\nloading: {model_name}...")
            model_dir = MODEL_ROOT / model_name
            if not model_dir.exists():
                raise FileNotFoundError(f"Model folder does not exist:{model_dir}")
            
            config_path = model_dir / config_filename
            model_path = model_dir / model_filename
            
            if not config_path.exists():
                raise FileNotFoundError(f"The configuration file does not exist:{config_path}")
            if not model_path.exists():
                raise FileNotFoundError(f"The configuration file does not exist:{model_path}")

            # 加载配置和模型
            config = utils.load_dict(str(model_dir))
            model = CompoundPred.model_initialize(**config)
            model.load_pretrained(str(model_path))

            # 确保配置中存在drug_encoding
            if "drug_encoding" not in config:
                config["drug_encoding"] = MODEL_DEFAULT_ENCODING.get(model_name, "rdkit_2d_normalized")
                print(f"Model {model_name} Configuration Supplement drug_encoding：{config['drug_encoding']}")

            self.models[model_name] = model
            self.configs[model_name] = config
            print(f"Model {model_name} loading Successfully")

            if self.default_model is None:
                self.default_model = model_name

        except Exception as e:
            raise RuntimeError(f"Model {model_name} loading failed: {e}")

    def get_model(self, model_name: Optional[str] = None) -> Any:
        if not self.models:
            raise RuntimeError("No models loaded")
        
        target_model_name = model_name or self.default_model
        if target_model_name not in self.models:
            raise ValueError(f"Model {target_model_name} does not exist,available models:{list(self.models.keys())}")
        return self.models[target_model_name]

    def get_model_config(self, model_name: Optional[str] = None) -> Any:
        if not self.configs:
            raise RuntimeError("No model configuration is loaded")
            
        target_model_name = model_name or self.default_model
        if target_model_name not in self.configs:
            raise ValueError(f"Model {target_model_name} the configuration does not exist")
        return self.configs[target_model_name]

    def list_models(self) -> List[str]:
        return list(self.models.keys())

# ========================== 模型加载 ==========================
try:
    print("初始化模型管理器...")
    model_manager = ModelManager()
    
    # 加载所有模型
    model_manager.load_model("rdkit_2d_normalizedModel", "config.pkl", "model.pt")
    model_manager.load_model("DaylightModel", "config.pkl", "model.pt")
    model_manager.load_model("ErGModel", "config.pkl", "model.pt")
    model_manager.load_model("MorganModel", "config.pkl", "model.pt")
    
    print(f"\nAll models loaded, available models: {model_manager.list_models()}")
    print(f"Default model:{model_manager.default_model}")

except Exception as e:
    raise RuntimeError(f"Model loading failed on startup:{e}")

# ========================== 内存数据库优化 ==========================
# 按模型分类存储结果，提高查询效率
db: Dict[str, dict] = {}  # 全局存储所有结果
model_results: Dict[str, List[str]] = {}  # 记录每个模型对应的结果ID列表

# 初始化每个模型的结果列表
for model_name in model_manager.list_models():
    model_results[model_name] = []

# ========================== 分子特征生成函数 ==========================
def smiles_to_rdkit2d(smiles_list: List[str]) -> np.ndarray:
    """将SMILES转换为RDKit 2D特征"""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol)
            ]
            features.append(desc)
        else:
            features.append([0.0]*5)
    return np.array(features, dtype=np.float32)

def smiles_to_morgan(smiles_list: List[str], radius=2, nBits=2048) -> np.ndarray:
    
    features = []
    # 处理RDKit版本差异（GetMorganGenerator/MorganGenerator）
    try:
        generator = AllChem.MorganGenerator(radius=radius, nBits=nBits)
    except AttributeError:
        generator = AllChem.GetMorganGenerator(radius=radius, nBits=nBits)
        
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((nBits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        else:
            features.append(np.zeros(nBits, dtype=np.float32))
    return np.array(features)

def smiles_to_daylight(smiles_list: List[str]) -> np.ndarray:
    """将SMILES转换为Daylight指纹"""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.RDKFingerprint(
                mol,
                maxPath=5,
                fpSize=2048,
                useHs=True,
                tgtDensity=0.0,
                minPath=1
            )
            arr = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            features.append(arr)
        else:
            features.append(np.zeros(2048, dtype=np.float32))
    return np.array(features)

def smiles_to_erg(smiles_list: List[str]) -> np.ndarray:
    """将SMILES转换为ErG特征"""
    features = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            features.append([
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                Descriptors.MolWt(mol)
            ])
        else:
            features.append([0.0, 0.0, 0.0])
    return np.array(features, dtype=np.float32)

# ========================== FastAPI 应用初始化 ==========================
app = FastAPI(title="pIC50 Predictor API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# ========================== 辅助函数 ==========================
def run_prediction(smiles_list: List[str], model_name: Optional[str] = None) -> List[ResultItem]:
    """执行预测并返回ResultItem列表"""
    if not smiles_list:
        return []
        
    # 筛选有效SMILES
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    invalid_smiles = [s for s in smiles_list if s not in valid_smiles]
    if invalid_smiles:
        print(f"Warning: Detected{len(invalid_smiles)}invalid SMILES, skipped")
    if not valid_smiles:
        raise HTTPException(status_code=400, detail="No valid SMILES string was provided.")
        
    try:
        # 获取模型和配置
        model = model_manager.get_model(model_name)
        model_config = model_manager.get_model_config(model_name)
        used_model = model_name or model_manager.default_model
        print(f"Using the Model {used_model} make predictions")

        # 获取编码方式
        drug_encoding = model_config.get(
            "drug_encoding", 
            MODEL_DEFAULT_ENCODING.get(used_model, "rdkit_2d_normalized")
        )
        print(f"Model encoding method: {drug_encoding}")

        # 构建输入数据（使用DeepPurpose标准处理流程）
        df_input = pd.DataFrame({
            "Drug": valid_smiles,
            "Label": [0.0] * len(valid_smiles)  # 占位标签
        })
        
        x_pred = utils.data_process(
            X_drug=df_input["Drug"].values,
            y=df_input["Label"].values,
            drug_encoding=drug_encoding,
            split_method="no_split"
        )

        # 执行预测
        predictions = model.predict(x_pred)
        
        # 统一预测结果格式为numpy数组
        if isinstance(predictions, list):
            predictions = np.array(predictions, dtype=np.float32).flatten()
        elif not isinstance(predictions, np.ndarray):
            raise TypeError(f"The predicted result type is abnormal: {type(predictions)}")

        print(f"The prediction is completed and the result shape is: {predictions.shape},Data Type: {predictions.dtype}")

        # 验证预测结果数量
        if len(predictions) != len(valid_smiles):
            raise ValueError(
                f"The number of prediction results does not match (input:{len(valid_smiles)},Output:{len(predictions)}）"
            )

        # 构建ResultItem列表
        current_time = datetime.datetime.now().isoformat()
        results = []
        for i, (smiles, pred_value) in enumerate(zip(valid_smiles, predictions)):
            try:
                mol = Chem.MolFromSmiles(smiles)
                pic50 = round(float(pred_value), 3)
                results.append(ResultItem(
                    id=str(uuid.uuid4()),
                    smiles=smiles,
                    pic50=pic50,
                    model_used=used_model,
                    timestamp=current_time,
                    molecule_name=f"Molecule-{len(db) + i + 1}",
                    mol_wt=Descriptors.MolWt(mol),
                    logp=Descriptors.MolLogP(mol),
                    hbd=Descriptors.NumHDonors(mol),
                    hba=Descriptors.NumHAcceptors(mol)
                ))
            except (TypeError, ValueError) as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Invalid predicted value '{pred_value}' (SMILES: {smiles}): {str(e)}"
                )

        return results

    except Exception as e:
        error_detail = f"Prediction failure: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

# ========================== API 接口定义 ==========================
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the pIC50 Prediction API",
        "available_models": model_manager.list_models(),
        "default_model": model_manager.default_model
    }

@app.post("/predict", response_model=List[ResultItem], tags=["predict"])
def predict_single(
    smiles: str = Query(..., description="SMILES string (supports multiple lines, one per line)"),
    model_name: Optional[str] = Query(None, description="Model name, optional value:" + ", ".join(model_manager.list_models()))
):
    """预测单个或多个SMILES的pIC50值"""
    smiles_list = [s.strip() for s in smiles.split("\n") if s.strip()]
    if not smiles_list:
        raise HTTPException(status_code=422, detail="Please provide at least one valid SMILES string.")
    
    results = run_prediction(smiles_list, model_name)
    
    # 保存结果到内存数据库，同时记录模型对应的结果ID
    for item in results:
        db[item.id] = item.dict()
        # 将结果ID添加到对应模型的列表中
        if item.model_used not in model_results:
            model_results[item.model_used] = []
        model_results[item.model_used].append(item.id)
        
    return results

@app.post("/upload_csv", response_model=List[ResultItem], tags=["predict"])
async def upload_csv(
    file: UploadFile = File(..., description="CSV file containing a SMILES column"),
    model_name: Optional[str] = Form(None, description="Model name, optional value:" + ", ".join(model_manager.list_models()))
):
    """从CSV文件上传SMILES并预测pIC50值"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type, please upload a CSV file.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        if 'SMILES' not in df.columns:
            raise HTTPException(status_code=400, detail="The CSV file must contain a 'SMILES' column.")
        smiles_list = df['SMILES'].dropna().tolist()
        print(f"Extract up to {len(smiles_list)} SMILES from CSV")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    results = run_prediction(smiles_list, model_name)
    
    # 保存结果到内存数据库，同时记录模型对应的结果ID
    for item in results:
        db[item.id] = item.dict()
        # 将结果ID添加到对应模型的列表中
        if item.model_used not in model_results:
            model_results[item.model_used] = []
        model_results[item.model_used].append(item.id)
        
    return results

@app.get("/results", response_model=PaginatedResults, tags=["result"])
def get_results(
    page: int = Query(1, ge=1, description="Page number, starting from 1"), 
    size: int = Query(10, ge=1, description="Number of items displayed per page"),
    model_name: Optional[str] = Query(None, description="Filter results by model name. Use 'all' to get all results."),
    sort_by: Optional[str] = Query("timestamp", description="Field to sort by (e.g., 'pic50', 'mol_wt', 'timestamp')"),
    sort_dir: Optional[str] = Query("desc", description="Sort direction: 'asc' for ascending, 'desc' for descending")
):
    """分页查询历史预测结果，支持按模型筛选和动态排序"""
    # 根据模型筛选结果
    if model_name and model_name.lower() != 'all':
        # 验证模型名称是否存在
        if model_name not in model_manager.list_models():
            raise HTTPException(status_code=400, detail=f"Model {model_name} does not exist")
        
        # 获取该模型的所有结果ID并获取对应的结果
        result_ids = model_results.get(model_name, [])
        all_items = [db[result_id] for result_id in result_ids if result_id in db]
    else:
        # 获取所有模型的结果
        all_items = list(db.values())
    
    # 动态排序逻辑
    if all_items:
        # 检查排序字段是否存在，以防出错
        sample_item = all_items[0]
        if sort_by not in sample_item:
            # 如果字段无效，则回退到默认按时间戳排序
            sort_by = "timestamp"
            
        reverse_order = sort_dir.lower() == "desc"
        
        # 对全量数据进行排序
        all_items.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse_order)
    
    # 计算分页信息
    total_items = len(all_items)
    total_pages = (total_items + size - 1) // size or 1
    start_index = (page - 1) * size
    paginated_items = all_items[start_index:start_index + size]
    
    return PaginatedResults(
        total_items=total_items,
        total_pages=total_pages,
        current_page=page,
        items=[ResultItem(**item) for item in paginated_items],
        model_used=model_name if model_name and model_name.lower() != 'all' else None
    )

@app.get("/plot_distribution", tags=["Visualization"])
def get_plot_distribution(model_name: Optional[str] = Query(None, description="Filter by model name")):
    """获取pIC50预测值的分布直方图数据"""
    if model_name:
        # 使用模型结果索引获取该模型的所有预测值
        result_ids = model_results.get(model_name, [])
        predictions = [db[result_id]['pic50'] for result_id in result_ids if result_id in db]
    else:
        predictions = [item['pic50'] for item in db.values()]

    if not predictions:
        return {"labels": [], "values": [], "model_used": model_name or "all"}
        
    hist, bin_edges = np.histogram(predictions, bins=10, range=(0, 10))
    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
    
    return {
        "labels": labels,
        "values": hist.tolist(),
        "model_used": model_name or "all"
    }

@app.get("/mol_image", tags=["Visualization"])
def get_mol_image(smiles: str = Query(..., description="SMILES string used to generate the structure image")):
    """生成分子结构的PNG图像"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string, unable to generate molecular structure.")
    
    img = Draw.MolToImage(mol, size=(250, 250))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")

@app.get("/models", tags=["Model Management"])
def get_available_models():
    """获取所有可用模型列表"""
    return {
        "available_models": model_manager.list_models(),
        "default_model": model_manager.default_model
    }
    