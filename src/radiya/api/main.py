from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import os
from pathlib import Path

# استيراد المسارات
from .routes import prediction, analysis, upload

app = FastAPI(
    title="Radiya - Customer Churn Prediction API",
    description="نظام التنبؤ بانسحاب العملاء باستخدام الذكاء الاصطناعي",
    version="1.0.0",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (إذا كانت موجودة)
static_path = Path(__file__).parent.parent / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# تضمين مسارات API
app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])

# صفحة رفع البيانات
@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    """صفحة رفع الملفات"""
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>رفع البيانات - رضية</title>
        <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            :root {
                --primary: #667eea;
                --secondary: #764ba2;
                --success: #10b981;
                --warning: #f59e0b;
                --danger: #ef4444;
                --dark: #1f2937;
                --gray: #6b7280;
                --light: #f9fafb;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Cairo', sans-serif;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .header h1 {
                color: var(--primary);
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .header p {
                color: var(--gray);
                font-size: 1.1rem;
            }
            
            .upload-area {
                border: 3px dashed var(--primary);
                border-radius: 15px;
                padding: 60px 20px;
                text-align: center;
                transition: all 0.3s ease;
                margin-bottom: 30px;
                cursor: pointer;
            }
            
            .upload-area:hover, .upload-area.dragover {
                border-color: var(--secondary);
                background: rgba(102, 126, 234, 0.05);
                transform: scale(1.02);
            }
            
            .upload-area i {
                font-size: 4rem;
                color: var(--primary);
                margin-bottom: 20px;
            }
            
            .upload-area h3 {
                color: var(--dark);
                margin-bottom: 10px;
                font-size: 1.3rem;
            }
            
            .upload-area p {
                color: var(--gray);
                margin-bottom: 20px;
            }
            
            .btn {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .file-info {
                background: var(--light);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
            }
            
            .file-info.show {
                display: block;
            }
            
            .progress-container {
                display: none;
                margin: 20px 0;
            }
            
            .progress-container.show {
                display: block;
            }
            
            .progress-bar {
                background: #e5e7eb;
                border-radius: 10px;
                height: 8px;
                margin: 15px 0;
                overflow: hidden;
            }
            
            .progress-fill {
                background: linear-gradient(90deg, var(--success), var(--primary));
                height: 100%;
                width: 0%;
                transition: width 0.3s ease;
            }
            
            .progress-info {
                display: flex;
                justify-content: space-between;
                align-items: center;
                color: var(--gray);
                font-size: 0.9rem;
            }
            
            .results {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background: var(--light);
                border-radius: 15px;
            }
            
            .results.show {
                display: block;
            }
            
            .alert {
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .alert-success {
                background: rgba(16, 185, 129, 0.1);
                color: var(--success);
                border: 1px solid rgba(16, 185, 129, 0.2);
            }
            
            .alert-error {
                background: rgba(239, 68, 68, 0.1);
                color: var(--danger);
                border: 1px solid rgba(239, 68, 68, 0.2);
            }
            
            .alert-warning {
                background: rgba(245, 158, 11, 0.1);
                color: var(--warning);
                border: 1px solid rgba(245, 158, 11, 0.2);
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .stat-value {
                font-size: 2rem;
                font-weight: 700;
                color: var(--primary);
                margin-bottom: 5px;
            }
            
            .stat-label {
                color: var(--gray);
                font-size: 0.9rem;
            }
            
            .hidden {
                display: none;
            }
            
            @media (max-width: 768px) {
                .container {
                    margin: 10px;
                    padding: 20px;
                }
                
                .upload-area {
                    padding: 40px 15px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1><i class="fas fa-cloud-upload-alt"></i> رفع البيانات</h1>
                <p>ارفع ملف البيانات الخاص بك وشاهد تحليل شامل للتنبؤ بانسحاب العملاء</p>
                <a href="/" style="color: var(--primary); text-decoration: none; margin-top: 10px; display: inline-block;">
                    <i class="fas fa-home"></i> العودة للرئيسية
                </a>
            </div>
            
            <div class="upload-area" id="uploadArea">
                <i class="fas fa-cloud-upload-alt"></i>
                <h3>اسحب الملف هنا أو اضغط للاختيار</h3>
                <p>JSON أو CSV - حتى 129 MB</p>
                <input type="file" id="fileInput" accept=".json,.csv" class="hidden">
                <button class="btn" onclick="document.getElementById('fileInput').click()">
                    <i class="fas fa-folder-open"></i>
                    اختر الملف
                </button>
            </div>
            
            <div class="file-info" id="fileInfo">
                <h4><i class="fas fa-file-alt"></i> معلومات الملف</h4>
                <p id="fileName"></p>
                <p id="fileSize"></p>
                <button class="btn" onclick="uploadFile()" id="uploadBtn">
                    <i class="fas fa-upload"></i>
                    رفع وتحليل
                </button>
                <button class="btn" onclick="resetUpload()" style="background: #6b7280; margin-right: 10px;">
                    <i class="fas fa-times"></i>
                    إلغاء
                </button>
            </div>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-info">
                    <span id="progressText">جاري الرفع...</span>
                    <span id="progressPercent">0%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div style="text-align: center; margin-top: 10px;">
                    <small id="currentStep">تحضير البيانات...</small>
                </div>
            </div>
            
            <div class="results" id="results">
                <h3><i class="fas fa-chart-bar"></i> نتائج التحليل</h3>
                <div id="alertContainer"></div>
                <div class="stats-grid" id="statsGrid"></div>
                <div id="detailedResults"></div>
                <div style="text-align: center; margin-top: 20px;">
                    <button class="btn" onclick="downloadResults()">
                        <i class="fas fa-download"></i>
                        تحميل التقرير
                    </button>
                    <button class="btn" onclick="resetUpload()" style="background: #6b7280; margin-right: 10px;">
                        <i class="fas fa-redo"></i>
                        رفع ملف جديد
                    </button>
                </div>
            </div>
        </div>
        
        <script>
            let selectedFile = null;
            let currentJobId = null;
            let pollingInterval = null;
            
            // إعداد drag and drop
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
            });
            
            uploadArea.addEventListener('drop', handleDrop, false);
            fileInput.addEventListener('change', handleFileSelect, false);
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            function handleDrop(e) {
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    selectFile(files[0]);
                }
            }
            
            function handleFileSelect(e) {
                if (e.target.files.length > 0) {
                    selectFile(e.target.files[0]);
                }
            }
            
            function selectFile(file) {
                const allowedTypes = ['.json', '.csv'];
                const fileName = file.name.toLowerCase();
                const isValidType = allowedTypes.some(type => fileName.endsWith(type));
                
                if (!isValidType) {
                    showAlert('نوع الملف غير مدعوم. يرجى رفع ملف JSON أو CSV', 'error');
                    return;
                }
                
                const fileSizeMB = file.size / (1024 * 1024);
                if (fileSizeMB > 129) {
                    showAlert(`حجم الملف كبير جداً (${fileSizeMB.toFixed(1)} MB). الحد الأقصى 129 MB`, 'error');
                    return;
                }
                
                selectedFile = file;
                
                document.getElementById('fileName').textContent = `الاسم: ${file.name}`;
                document.getElementById('fileSize').textContent = `الحجم: ${formatFileSize(file.size)}`;
                document.getElementById('fileInfo').classList.add('show');
                
                showAlert(`تم اختيار الملف: ${file.name}`, 'success');
            }
            
            async function uploadFile() {
                if (!selectedFile) {
                    showAlert('يرجى اختيار ملف أولاً', 'warning');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', selectedFile);
                
                try {
                    document.getElementById('progressContainer').classList.add('show');
                    document.getElementById('fileInfo').classList.remove('show');
                    
                    updateProgress(0, 'جاري رفع الملف...', 'رفع الملف...');
                    
                    const response = await fetch('/api/v1/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(result.detail || 'خطأ في رفع الملف');
                    }
                    
                    currentJobId = result.job_id;
                    showAlert(`تم رفع الملف بنجاح! جاري المعالجة...`, 'success');
                    
                    startPolling();
                    
                } catch (error) {
                    console.error('خطأ في الرفع:', error);
                    showAlert(`خطأ في رفع الملف: ${error.message}`, 'error');
                    document.getElementById('progressContainer').classList.remove('show');
                    document.getElementById('fileInfo').classList.add('show');
                }
            }
            
            function startPolling() {
                if (pollingInterval) {
                    clearInterval(pollingInterval);
                }
                
                pollingInterval = setInterval(async () => {
                    try {
                        const response = await fetch(`/api/v1/status/${currentJobId}`);
                        const status = await response.json();
                        
                        updateProgress(status.progress, status.current_step, `${status.progress}%`);
                        
                        if (status.status === 'completed') {
                            clearInterval(pollingInterval);
                            showResults(status.results);
                            showAlert('تم تحليل البيانات بنجاح!', 'success');
                        } else if (status.status === 'error') {
                            clearInterval(pollingInterval);
                            showAlert(`خطأ في المعالجة: ${status.error}`, 'error');
                            document.getElementById('progressContainer').classList.remove('show');
                        }
                        
                    } catch (error) {
                        console.error('خطأ في مراقبة التقدم:', error);
                        clearInterval(pollingInterval);
                        showAlert('خطأ في متابعة حالة المعالجة', 'error');
                    }
                }, 2000);
            }
            
            function updateProgress(percent, step, percentText) {
                document.getElementById('progressFill').style.width = percent + '%';
                document.getElementById('progressPercent').textContent = percentText;
                document.getElementById('currentStep').textContent = step;
            }
            
            function showResults(results) {
                document.getElementById('progressContainer').classList.remove('show');
                
                const statsHtml = `
                    <div class="stat-card">
                        <div class="stat-value">${results.data_summary.total_records.toLocaleString()}</div>
                        <div class="stat-label">إجمالي السجلات</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${results.data_summary.unique_users.toLocaleString()}</div>
                        <div class="stat-label">مستخدم فريد</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${results.data_summary.churn_rates.combined}%</div>
                        <div class="stat-label">معدل الانسحاب</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${Math.max(...Object.values(results.best_models).map(m => Math.round(m.auc_score * 100)))}%</div>
                        <div class="stat-label">أفضل دقة</div>
                    </div>
                `;
                
                document.getElementById('statsGrid').innerHTML = statsHtml;
                
                let alertsHtml = '';
                results.recommendations.forEach(rec => {
                    const alertType = rec.type === 'urgent' ? 'error' : rec.type === 'warning' ? 'warning' : 'success';
                    alertsHtml += `
                        <div class="alert alert-${alertType}">
                            <i class="fas fa-${rec.type === 'urgent' ? 'exclamation-triangle' : rec.type === 'warning' ? 'exclamation-circle' : 'check-circle'}"></i>
                            <div>
                                <strong>${rec.title}</strong><br>
                                <small>${rec.description}</small>
                            </div>
                        </div>
                    `;
                });
                
                document.getElementById('alertContainer').innerHTML = alertsHtml;
                
                let modelsHtml = '<h4>أداء النماذج:</h4><div style="display: grid; gap: 15px;">';
                Object.entries(results.best_models).forEach(([method, model]) => {
                    modelsHtml += `
                        <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid var(--primary);">
                            <strong>طريقة ${method}:</strong> ${model.model_name}<br>
                            <small>دقة: ${Math.round(model.auc_score * 100)}% | دقة التصنيف: ${Math.round(model.precision * 100)}%</small>
                        </div>
                    `;
                });
                modelsHtml += '</div>';
                
                document.getElementById('detailedResults').innerHTML = modelsHtml;
                document.getElementById('results').classList.add('show');
                
                window.analysisResults = results;
            }
            
            function showAlert(message, type = 'success') {
                const alertContainer = document.getElementById('alertContainer');
                const iconMap = {
                    success: 'check-circle',
                    error: 'times-circle',
                    warning: 'exclamation-triangle'
                };
                
                const alert = document.createElement('div');
                alert.className = `alert alert-${type}`;
                alert.innerHTML = `
                    <i class="fas fa-${iconMap[type]}"></i>
                    <span>${message}</span>
                `;
                
                alertContainer.appendChild(alert);
                
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.parentNode.removeChild(alert);
                    }
                }, 5000);
            }
            
            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            function resetUpload() {
                selectedFile = null;
                currentJobId = null;
                
                if (pollingInterval) {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                }
                
                document.getElementById('fileInfo').classList.remove('show');
                document.getElementById('progressContainer').classList.remove('show');
                document.getElementById('results').classList.remove('show');
                document.getElementById('fileInput').value = '';
                document.getElementById('alertContainer').innerHTML = '';
                
                showAlert('تم إعادة تعيين النموذج', 'success');
            }
            
            function downloadResults() {
                if (window.analysisResults) {
                    const dataStr = JSON.stringify(window.analysisResults, null, 2);
                    const dataBlob = new Blob([dataStr], {type: 'application/json'});
                    const url = URL.createObjectURL(dataBlob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = `radiya_analysis_${new Date().toISOString().slice(0, 19)}.json`;
                    link.click();
                    URL.revokeObjectURL(url);
                    
                    showAlert('تم تحميل التقرير بنجاح', 'success');
                } else {
                    showAlert('لا توجد نتائج للتحميل', 'warning');
                }
            }
            
            document.addEventListener('DOMContentLoaded', () => {
                showAlert('مرحباً بك في نظام رضية! ارفع ملف البيانات للبدء', 'success');
            });
        </script>
    </body>
    </html>
    """

# الصفحة الرئيسية المحدثة
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>رضية - نظام التنبؤ بانسحاب العملاء</title>
        <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body {
                font-family: 'Cairo', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 40px 20px;
                min-height: 100vh;
                color: white;
                text-align: center;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                padding: 60px 40px;
                border-radius: 25px;
                backdrop-filter: blur(15px);
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            }
            h1 {
                font-size: 4rem;
                margin-bottom: 20px;
                font-weight: 900;
            }
            .subtitle {
                font-size: 1.5rem;
                margin-bottom: 40px;
                opacity: 0.9;
            }
            .icon {
                font-size: 5rem;
                margin-bottom: 30px;
                background: linear-gradient(45deg, #f093fb, #f5af19);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 30px;
                margin: 60px 0;
            }
            .feature {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 20px;
                transition: all 0.3s ease;
                border: 1px solid rgba(255,255,255,0.2);
            }
            .feature:hover {
                transform: translateY(-10px);
                background: rgba(255,255,255,0.15);
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            }
            .feature h3 {
                font-size: 1.4rem;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
            }
            .btn {
                background: rgba(255,255,255,0.2);
                color: white;
                padding: 18px 35px;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 15px;
                font-size: 1.2rem;
                font-weight: 600;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 12px;
                margin: 15px;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            .btn:hover {
                background: rgba(255,255,255,0.3);
                transform: translateY(-3px);
                box-shadow: 0 15px 30px rgba(0,0,0,0.2);
                border-color: rgba(255,255,255,0.5);
            }
            .btn-primary {
                background: linear-gradient(135deg, #f093fb, #f5af19);
                border: none;
                font-size: 1.3rem;
                padding: 20px 40px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 20px;
                margin: 40px 0;
            }
            .stat {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 15px;
            }
            .stat-number {
                font-size: 2.5rem;
                font-weight: 900;
                display: block;
            }
            .stat-label {
                opacity: 0.8;
                margin-top: 5px;
            }
            @media (max-width: 768px) {
                .container {
                    padding: 30px 20px;
                }
                h1 {
                    font-size: 2.5rem;
                }
                .features {
                    grid-template-columns: 1fr;
                    gap: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">
                <i class="fas fa-brain"></i>
            </div>
            <h1>رضية</h1>
            <p class="subtitle">نظام التنبؤ بانسحاب العملاء باستخدام الذكاء الاصطناعي</p>
            
            <div class="stats">
                <div class="stat">
                    <span class="stat-number">87%</span>
                    <div class="stat-label">دقة التنبؤ</div>
                </div>
                <div class="stat">
                    <span class="stat-number">6</span>
                    <div class="stat-label">نماذج ذكية</div>
                </div>
                <div class="stat">
                    <span class="stat-number">129MB</span>
                    <div class="stat-label">حجم الملف الأقصى</div>
                </div>
            </div>
            
            <div class="features">
                <div class="feature">
                    <h3><i class="fas fa-upload"></i> رفع البيانات</h3>
                    <p>ارفع ملفات JSON أو CSV حتى 129 MB وشاهد التحليل الشامل للبيانات</p>
                </div>
                <div class="feature">
                    <h3><i class="fas fa-chart-line"></i> تحليل متقدم</h3>
                    <p>6 خوارزميات مختلفة تحلل بياناتك وتتنبأ بسلوك العملاء</p>
                </div>
                <div class="feature">
                    <h3><i class="fas fa-download"></i> تقارير شاملة</h3>
                    <p>احصل على تقارير مفصلة مع توصيات عملية للاحتفاظ بالعملاء</p>
                </div>
            </div>
            
            <div>
                <a href="/upload" class="btn btn-primary">
                    <i class="fas fa-rocket"></i>
                    ابدأ التحليل الآن
                </a>
                <a href="/docs" class="btn">
                    <i class="fas fa-book"></i>
                    وثائق API
                </a>
                <a href="/api/v1/analysis/summary" class="btn">
                    <i class="fas fa-chart-bar"></i>
                    ملخص النشاط
                </a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "radiya-api", "upload_enabled": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)