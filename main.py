"""
=============================================================================
 Network Anomaly Detection - REST API
 Automated Monitoring & Alert System
 Built with FastAPI for Google Cloud Run
=============================================================================
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
import os
import json
import sqlite3
import csv
import io
from datetime import datetime

# ============================================================================
# APP SETUP
# ============================================================================
app = FastAPI(
    title="Network Anomaly Detection API",
    description="AI-powered network monitoring — detects anomalies automatically",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATABASE
# ============================================================================
DB_PATH = "network_monitoring.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS monitoring_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        server_id TEXT,
        cpu_usage REAL,
        memory_usage REAL,
        disk_io REAL,
        network_traffic REAL,
        error_count INTEGER,
        response_time REAL,
        active_connections INTEGER,
        packet_loss REAL,
        is_anomaly BOOLEAN,
        anomaly_score REAL,
        risk_level TEXT,
        timestamp TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        server_id TEXT,
        risk_level TEXT,
        anomaly_score REAL,
        details TEXT,
        timestamp TEXT,
        acknowledged BOOLEAN DEFAULT 0
    )''')
    conn.commit()
    conn.close()

init_db()

# ============================================================================
# LOAD MODEL
# ============================================================================
MODEL_DIR = os.getenv("MODEL_DIR", "outputs/models")
REPORT_DIR = os.getenv("REPORT_DIR", "outputs/reports")
DATA_DIR = os.getenv("DATA_DIR", "data")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "anomaly_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    model = None
    scaler = None

metadata = {}
try:
    with open(os.path.join(REPORT_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
except Exception:
    pass

# In-memory database for alerts history
alerts_history = []
monitoring_log = []

FEATURES = ['cpu_usage', 'memory_usage', 'disk_io', 'network_traffic',
            'error_count', 'response_time', 'active_connections', 'packet_loss']

# ============================================================================
# SCHEMAS
# ============================================================================
class ServerMetrics(BaseModel):
    server_id: str = Field(..., description="Server identifier", json_schema_extra={"examples": ["SRV-001"]})
    cpu_usage: float = Field(..., description="CPU usage percentage", ge=0, le=100)
    memory_usage: float = Field(..., description="Memory usage percentage", ge=0, le=100)
    disk_io: float = Field(..., description="Disk I/O in MB/s", ge=0)
    network_traffic: float = Field(..., description="Network traffic in Mbps", ge=0)
    error_count: int = Field(0, description="Number of errors", ge=0)
    response_time: float = Field(..., description="Response time in ms", ge=0)
    active_connections: int = Field(0, description="Active connections count", ge=0)
    packet_loss: float = Field(0.0, description="Packet loss percentage", ge=0, le=100)

class BatchMetrics(BaseModel):
    servers: List[ServerMetrics]

class AnomalyResult(BaseModel):
    server_id: str
    is_anomaly: bool
    anomaly_score: float
    risk_level: str
    timestamp: str
    details: dict

class BatchAnomalyResult(BaseModel):
    results: List[AnomalyResult]
    total_servers: int
    anomalies_found: int
    healthy_servers: int

class AlertRecord(BaseModel):
    server_id: str
    risk_level: str
    anomaly_score: float
    timestamp: str
    metrics: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    total_alerts: int
    servers_monitored: int

# ============================================================================
# PREDICTION LOGIC
# ============================================================================
def analyze_server(metrics: ServerMetrics) -> AnomalyResult:
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array([[
        metrics.cpu_usage,
        metrics.memory_usage,
        metrics.disk_io,
        metrics.network_traffic,
        metrics.error_count,
        metrics.response_time,
        metrics.active_connections,
        metrics.packet_loss
    ]])

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]

    is_anomaly = prediction == -1
    normalized_score = round(max(0, min(1, 0.5 - score)), 4)

    if normalized_score > 0.7:
        risk_level = "CRITICAL"
    elif normalized_score > 0.5:
        risk_level = "HIGH"
    elif normalized_score > 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    details = {
        "cpu_status": "HIGH" if metrics.cpu_usage > 85 else "NORMAL",
        "memory_status": "HIGH" if metrics.memory_usage > 85 else "NORMAL",
        "network_status": "HIGH" if metrics.network_traffic > 500 else "NORMAL",
        "error_status": "HIGH" if metrics.error_count > 10 else "NORMAL",
        "response_status": "SLOW" if metrics.response_time > 500 else "NORMAL",
        "packet_loss_status": "HIGH" if metrics.packet_loss > 5 else "NORMAL",
    }

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    result = AnomalyResult(
        server_id=metrics.server_id,
        is_anomaly=is_anomaly,
        anomaly_score=normalized_score,
        risk_level=risk_level if is_anomaly else "HEALTHY",
        timestamp=timestamp,
        details=details
    )

    # Save to in-memory log
    monitoring_log.append({
        "server_id": metrics.server_id,
        "is_anomaly": is_anomaly,
        "risk_level": result.risk_level,
        "score": normalized_score,
        "timestamp": timestamp
    })

    # Save to SQLite database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO monitoring_logs
        (server_id, cpu_usage, memory_usage, disk_io, network_traffic,
         error_count, response_time, active_connections, packet_loss,
         is_anomaly, anomaly_score, risk_level, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (metrics.server_id, metrics.cpu_usage, metrics.memory_usage,
         metrics.disk_io, metrics.network_traffic, metrics.error_count,
         metrics.response_time, metrics.active_connections, metrics.packet_loss,
         is_anomaly, normalized_score, result.risk_level, timestamp))

    if is_anomaly:
        alert = {
            "server_id": metrics.server_id,
            "risk_level": result.risk_level,
            "anomaly_score": normalized_score,
            "timestamp": timestamp,
            "metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "disk_io": metrics.disk_io,
                "network_traffic": metrics.network_traffic,
                "error_count": metrics.error_count,
                "response_time": metrics.response_time,
                "active_connections": metrics.active_connections,
                "packet_loss": metrics.packet_loss
            }
        }
        alerts_history.append(alert)

        c.execute("""INSERT INTO alerts
            (server_id, risk_level, anomaly_score, details, timestamp)
            VALUES (?, ?, ?, ?, ?)""",
            (metrics.server_id, result.risk_level, normalized_score,
             json.dumps(details), timestamp))

    conn.commit()
    conn.close()

    return result

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    return HealthResponse(
        status="online",
        model_loaded=model is not None,
        model_type=metadata.get("model"),
        total_alerts=len(alerts_history),
        servers_monitored=len(set(log["server_id"] for log in monitoring_log))
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type=metadata.get("model"),
        total_alerts=len(alerts_history),
        servers_monitored=len(set(log["server_id"] for log in monitoring_log))
    )

@app.post("/analyze", response_model=AnomalyResult, tags=["Anomaly Detection"])
async def analyze(metrics: ServerMetrics):
    """
    Analyze a single server's metrics for anomalies.
    Returns anomaly status, risk level, and detailed breakdown.
    """
    return analyze_server(metrics)

@app.post("/analyze/batch", response_model=BatchAnomalyResult, tags=["Anomaly Detection"])
async def analyze_batch(batch: BatchMetrics):
    """
    Analyze multiple servers at once.
    UiPath uses this for batch monitoring.
    """
    results = [analyze_server(s) for s in batch.servers]
    anomalies = [r for r in results if r.is_anomaly]

    return BatchAnomalyResult(
        results=results,
        total_servers=len(results),
        anomalies_found=len(anomalies),
        healthy_servers=len(results) - len(anomalies)
    )

@app.post("/analyze/file", tags=["Anomaly Detection"])
async def analyze_file(file: UploadFile = File(...)):
    """Upload a CSV file to analyze multiple servers at once."""
    content = await file.read()
    decoded = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))

    results = []
    for row in reader:
        try:
            metrics = ServerMetrics(
                server_id=row["server_id"],
                cpu_usage=float(row["cpu_usage"]),
                memory_usage=float(row["memory_usage"]),
                disk_io=float(row["disk_io"]),
                network_traffic=float(row["network_traffic"]),
                error_count=int(float(row["error_count"])),
                response_time=float(row["response_time"]),
                active_connections=int(float(row["active_connections"])),
                packet_loss=float(row["packet_loss"])
            )
            result = analyze_server(metrics)
            results.append({
                "server_id": result.server_id,
                "is_anomaly": result.is_anomaly,
                "anomaly_score": result.anomaly_score,
                "risk_level": result.risk_level,
                "timestamp": result.timestamp,
                "details": result.details
            })
        except Exception as e:
            results.append({"server_id": row.get("server_id", "unknown"), "error": str(e)})

    anomalies = [r for r in results if r.get("is_anomaly", False)]
    return {
        "filename": file.filename,
        "total_servers": len(results),
        "anomalies_found": len(anomalies),
        "healthy_servers": len(results) - len(anomalies),
        "results": results
    }

@app.get("/alerts", tags=["Alerts"])
async def get_alerts(limit: int = 50):
    """Get recent alerts history"""
    return {
        "total_alerts": len(alerts_history),
        "alerts": alerts_history[-limit:]
    }

@app.get("/alerts/critical", tags=["Alerts"])
async def get_critical_alerts():
    """Get only CRITICAL and HIGH risk alerts"""
    critical = [a for a in alerts_history if a["risk_level"] in ["CRITICAL", "HIGH"]]
    return {
        "total": len(critical),
        "alerts": critical
    }

@app.get("/dashboard/summary", tags=["Dashboard"])
async def dashboard_summary():
    """Get monitoring summary for dashboard"""
    if not monitoring_log:
        return {"message": "No data yet. Send server metrics to /analyze first."}

    df_log = pd.DataFrame(monitoring_log)

    total = len(df_log)
    anomalies = len(df_log[df_log["is_anomaly"] == True])
    servers = df_log["server_id"].nunique()

    risk_counts = df_log["risk_level"].value_counts().to_dict()

    return {
        "total_checks": total,
        "total_anomalies": anomalies,
        "anomaly_rate": round(anomalies / total * 100, 2) if total > 0 else 0,
        "servers_monitored": servers,
        "risk_breakdown": risk_counts,
        "last_check": monitoring_log[-1]["timestamp"] if monitoring_log else None
    }

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model metadata and performance metrics"""
    if not metadata:
        raise HTTPException(status_code=404, detail="No metadata available")
    return metadata

@app.delete("/alerts/clear", tags=["Alerts"])
async def clear_alerts():
    """Clear all alerts history"""
    alerts_history.clear()
    monitoring_log.clear()
    return {"message": "All alerts cleared", "status": "ok"}

# ============================================================================
# DATABASE ENDPOINTS
# ============================================================================

@app.get("/db/logs", tags=["Database"])
async def get_db_logs(limit: int = 100):
    """Returns last N monitoring logs from database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM monitoring_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"total": len(rows), "logs": rows}

@app.get("/db/alerts", tags=["Database"])
async def get_db_alerts():
    """Returns all unacknowledged alerts from database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY id DESC")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"total": len(rows), "alerts": rows}

@app.get("/db/stats", tags=["Database"])
async def get_db_stats():
    """Returns aggregate stats from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM monitoring_logs")
    total_logs = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM alerts")
    total_alerts = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT server_id) FROM monitoring_logs")
    unique_servers = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM monitoring_logs WHERE is_anomaly = 1")
    total_anomalies = c.fetchone()[0]

    anomaly_rate = round(total_anomalies / total_logs * 100, 2) if total_logs > 0 else 0

    conn.close()
    return {
        "total_logs": total_logs,
        "total_alerts": total_alerts,
        "unique_servers": unique_servers,
        "anomaly_rate": anomaly_rate
    }

@app.post("/db/acknowledge/{alert_id}", tags=["Database"])
async def acknowledge_alert(alert_id: int):
    """Marks an alert as acknowledged"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    if c.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Alert not found")
    conn.commit()
    conn.close()
    return {"message": f"Alert {alert_id} acknowledged", "status": "ok"}

# ============================================================================
# REPORT
# ============================================================================

@app.get("/report/generate", tags=["Reports"])
async def generate_report():
    """Generate automated monitoring report"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM monitoring_logs")
    total_checks = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM monitoring_logs WHERE is_anomaly = 1")
    total_anomalies = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT server_id) FROM monitoring_logs")
    unique_servers = c.fetchone()[0]

    c.execute("""SELECT server_id, COUNT(*) as count FROM monitoring_logs
                 WHERE is_anomaly = 1 GROUP BY server_id ORDER BY count DESC LIMIT 5""")
    top_problematic = [{"server_id": row[0], "anomaly_count": row[1]} for row in c.fetchall()]

    c.execute("SELECT risk_level, COUNT(*) FROM alerts GROUP BY risk_level")
    risk_breakdown = {row[0]: row[1] for row in c.fetchall()}

    c.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0")
    unacknowledged = c.fetchone()[0]

    conn.close()

    return {
        "report_title": "Network Monitoring Report",
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "summary": {
            "total_checks": total_checks,
            "total_anomalies": total_anomalies,
            "anomaly_rate": round(total_anomalies / total_checks * 100, 2) if total_checks > 0 else 0,
            "unique_servers": unique_servers,
            "unacknowledged_alerts": unacknowledged
        },
        "top_problematic_servers": top_problematic,
        "risk_breakdown": risk_breakdown,
        "model_info": metadata
    }

# ============================================================================
# DASHBOARD
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard_page():
    """Serve the monitoring dashboard"""
    with open("dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
