import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import {
  UploadCloud,
  Droplets,
  Microscope,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Download,
  RotateCcw,
  Sliders,
  Sparkles,
  Info,
  Maximize2,
  Minimize2,
  CheckCircle2,
  Activity,
  Cpu,
  Layers,
  ArrowRight
} from 'lucide-react';
import './App.css';

const API_BASE = ''; // Uses Vite proxy to http://127.0.0.1:8000

export default function App() {
  // System / Health State
  const [health, setHealth] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);

  // Model Parameters
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [showSettings, setShowSettings] = useState(false);

  // Workflow & Results State
  const [loading, setLoading] = useState(false);
  const [loadingTime, setLoadingTime] = useState(0);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('annotated'); // 'annotated' | 'original'
  const [isZoomed, setIsZoomed] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // File input ref
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  // Check Backend Health and fetch curated samples on load
  useEffect(() => {
    fetchHealth();
    fetchSamples();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth({ status: 'offline' });
      }
    } catch {
      setHealth({ status: 'offline' });
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data);
      }
    } catch (err) {
      console.error('Failed to fetch samples:', err);
    }
  };

  // Timer while loading
  useEffect(() => {
    let interval;
    if (loading) {
      setLoadingTime(0);
      interval = setInterval(() => {
        setLoadingTime((prev) => +(prev + 0.1).toFixed(1));
      }, 100);
    }
    return () => clearInterval(interval);
  }, [loading]);

  // Handle sample detection
  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    setLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/sample/${sample.id}?confidence=${confThreshold}&iou=${iouThreshold}`
      );
      if (!res.ok) throw new Error('Sample detection failed.');
      const data = await res.json();
      setResults(data);
      confetti({ particleCount: 60, spread: 70, origin: { y: 0.7 } });
    } catch (err) {
      setErrorMessage(err.message || 'Error processing sample');
    } finally {
      setLoading(false);
    }
  };

  // Handle local file upload
  const handleFileUpload = async (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setErrorMessage('Please provide a valid microscopic image file (JPG, PNG, TIFF).');
      return;
    }
    setSelectedSample(null);
    setLoading(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', confThreshold);
    formData.append('iou', iouThreshold);

    try {
      const res = await fetch(`${API_BASE}/api/detect`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Inference failed.');
      }
      const data = await res.json();
      setResults(data);
      confetti({ particleCount: 70, spread: 70, origin: { y: 0.7 } });
    } catch (err) {
      setErrorMessage(err.message || 'Error detecting image');
    } finally {
      setLoading(false);
    }
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  // Re-run detection when parameters change
  const handleRerunWithNewParams = () => {
    if (selectedSample) {
      handleSelectSample(selectedSample);
    }
  };

  // Reset analysis
  const handleReset = () => {
    setResults(null);
    setSelectedSample(null);
    setErrorMessage(null);
  };

  // Download Analysis Audit Report
  const handleDownloadReport = () => {
    if (!results) return;

    const reportContent = `========================================================================
       MICROPLASTIC CONTAMINATION & PARTICLE CHARACTERIZATION AUDIT REPORT
========================================================================
Date & Time:         ${new Date().toLocaleString()}
Specimen Sample:     ${results.filename || 'Uploaded Water Specimen'}
Image Dimensions:    ${results.image_dimensions?.width} x ${results.image_dimensions?.height} px
AI Architecture:     YOLOv8n (Fine-tuned, 800px input resolution)
Model Weights:       models/exp2_long_training/weights/best.pt
Inference Hardware:  ${health?.gpu_name || 'NVIDIA GPU / CUDA'}
Confidence Thresh:   ${confThreshold}
IoU NMS Thresh:      ${iouThreshold}

------------------------------------------------------------------------
SUMMARY OF FINDINGS
------------------------------------------------------------------------
Microplastics Count:        ${results.count} detected particles
Contamination Severity:     ${results.contamination.level.toUpperCase()}
Risk Assessment:            ${results.contamination.desc}
Mean Detection Confidence:  ${results.average_confidence}%
Total Particle Area:        ${results.total_area_px.toLocaleString()} px²
Processing Latency:         ${results.inference_time_ms} ms

------------------------------------------------------------------------
INDIVIDUAL PARTICLE BREAKDOWN
------------------------------------------------------------------------
${
  results.particles.length === 0
    ? 'No microplastic particles detected at threshold ' + confThreshold
    : results.particles
        .map(
          (p) =>
            `#${String(p.id).padEnd(3)} | Confidence: ${p.confidence}% | Area: ${String(
              p.area_px
            ).padStart(5)} px² | Bounding Box [x1, y1, x2, y2]: [${p.bbox.join(', ')}]`
        )
        .join('\n')
}

------------------------------------------------------------------------
Recommendation:
${
  results.count === 0
    ? 'Water sample indicates clean purity with no detected particulate debris.'
    : results.count <= 3
    ? 'Low trace presence. Periodic monitoring of source suggested.'
    : results.count <= 7
    ? 'Moderate contamination. Standard membrane micro-filtration recommended.'
    : 'Severe microplastic pollution. Water unsuitable for consumption without high-efficiency reverse osmosis or ultra-filtration treatment.'
}
========================================================================
Generated by Microplastic Detection AI System
========================================================================
`;

    const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `microplastic_report_${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Download Annotated Image
  const handleDownloadImage = () => {
    if (!results || !results.annotated_image) return;
    const link = document.createElement('a');
    link.href = results.annotated_image;
    link.download = `annotated_${results.filename || 'microplastic_detection'}.jpg`;
    link.click();
  };

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Droplets className="icon-cyan" size={24} />
          </div>
          <div>
            <h1 className="brand-title">MICROPLASTIC DETECTION AI</h1>
            <p className="brand-sub">Computer Vision Water Quality & Microplastic Particle Analysis</p>
          </div>
        </div>

        <div className="nav-actions">
          {/* Hardware & System Status Pill */}
          <div className={`status-pill ${health?.status === 'online' ? 'online' : 'offline'}`}>
            <Cpu size={14} />
            <span>
              {health?.status === 'online'
                ? `${health.gpu_name || 'GPU Device 0'} (Online)`
                : 'Backend Disconnected'}
            </span>
          </div>

          {/* Model info chip */}
          <div className="model-pill">
            <Layers size={14} />
            <span>YOLOv8n • 800px</span>
          </div>

          {/* Settings toggle button */}
          <button
            className={`icon-btn ${showSettings ? 'active' : ''}`}
            onClick={() => setShowSettings(!showSettings)}
            title="Detection Settings"
          >
            <Sliders size={18} />
          </button>
        </div>
      </header>

      {/* Settings Drawer */}
      {showSettings && (
        <section className="settings-panel">
          <div className="settings-header">
            <div className="flex items-center gap-2">
              <Sliders size={18} className="text-cyan" />
              <h3>Inference Detection Parameters</h3>
            </div>
            <button
              className="reset-params-btn"
              onClick={() => {
                setConfThreshold(0.25);
                setIouThreshold(0.45);
              }}
            >
              Reset to Optimal (Conf 0.25, IoU 0.45)
            </button>
          </div>

          <div className="settings-grid">
            <div className="setting-item">
              <div className="setting-label">
                <span>Confidence Threshold (conf)</span>
                <span className="setting-value">{confThreshold}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.80"
                step="0.05"
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              />
              <span className="setting-hint">Optimal F1 operating point: 0.25</span>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span>NMS IoU Threshold (iou)</span>
                <span className="setting-value">{iouThreshold}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.80"
                step="0.05"
                value={iouThreshold}
                onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
              />
              <span className="setting-hint">Suppresses duplicate boxes on elongated fibers</span>
            </div>
          </div>

          {results && (
            <div className="settings-footer">
              <button className="reanalyze-btn" onClick={handleRerunWithNewParams}>
                Re-run with new thresholds <ArrowRight size={16} />
              </button>
            </div>
          )}
        </section>
      )}

      {/* Main Content Area */}
      <main className="main-content">
        {errorMessage && (
          <div className="error-banner">
            <AlertTriangle size={20} />
            <span>{errorMessage}</span>
            <button onClick={() => setErrorMessage(null)}>×</button>
          </div>
        )}

        {!results && !loading && (
          <div className="upload-view">
            {/* Hero Text */}
            <div className="hero-banner">
              <h2>Detect and analyze microplastics in water sample images</h2>
              <p>
                Utilizes high-resolution YOLOv8n deep learning to accurately segment, count, and measure
                microscopic polymers, fibers, and fragments in aquatic specimens.
              </p>
            </div>

            {/* Drag & Drop Upload Zone */}
            <div
              className={`dropzone ${isDragging ? 'dragging' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: 'none' }}
                accept="image/*"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
              <div className="dropzone-icon-wrap">
                <UploadCloud size={48} className="dropzone-icon" />
              </div>
              <h3>Drop your water sample image here, or browse</h3>
              <p>Supports high-resolution microscopic imagery (JPG, PNG, TIFF, BMP)</p>
              <button
                type="button"
                className="upload-trigger-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current.click();
                }}
              >
                Upload Image
              </button>
            </div>

            {/* Quick-Select Curated Water Samples */}
            {samples.length > 0 && (
              <div className="sample-gallery-section">
                <div className="section-title-wrap">
                  <Microscope size={18} className="text-cyan" />
                  <h4>Quick Test: Curated Water Specimen Gallery</h4>
                  <span className="sample-badge">Click to analyze instantly</span>
                </div>
                <div className="sample-grid">
                  {samples.map((sample, idx) => (
                    <button
                      key={sample.id}
                      className="sample-card"
                      onClick={() => handleSelectSample(sample)}
                    >
                      <div className="sample-preview">
                        <Droplets size={22} className="sample-drop-icon" />
                      </div>
                      <div className="sample-meta">
                        <span className="sample-title">Water Specimen #{idx + 1}</span>
                        <span className="sample-filename">{sample.display_name}</span>
                      </div>
                      <ArrowRight size={16} className="sample-arrow" />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Loading / Radar Scan State */}
        {loading && (
          <div className="loading-state">
            <div className="radar-container">
              <div className="radar-scanner"></div>
              <div className="radar-rings"></div>
              <Microscope size={44} className="radar-center-icon" />
            </div>
            <h3>Scanning Water Sample...</h3>
            <p>Running YOLOv8n deep neural detection on {loadingTime}s</p>
            <div className="loading-steps">
              <span>Resolving microscopic boundaries</span> • <span>Classifying particulate matter</span> • <span>Calculating pixel area</span>
            </div>
          </div>
        )}

        {/* Results Dashboard */}
        {results && !loading && (
          <div className="results-container">
            {/* Top Bar: Back button & Sample filename */}
            <div className="results-header">
              <button className="back-btn" onClick={handleReset}>
                <RotateCcw size={16} />
                <span>Analyze Another Sample</span>
              </button>
              <div className="results-meta">
                <span className="results-specimen">Specimen: <strong>{results.filename}</strong></span>
                <span className="results-dims">{results.image_dimensions?.width} × {results.image_dimensions?.height} px</span>
              </div>
            </div>

            {/* Core KPI Metrics Grid */}
            <div className="metrics-grid">
              {/* Microplastics Count */}
              <div className="metric-card cyan-glow">
                <div className="metric-icon-wrap cyan">
                  <Microscope size={24} />
                </div>
                <div className="metric-content">
                  <span className="metric-label">Microplastics</span>
                  <div className="metric-value-row">
                    <span className="metric-number">{results.count}</span>
                    <span className="metric-unit">particles</span>
                  </div>
                </div>
              </div>

              {/* Avg Confidence */}
              <div className="metric-card blue-glow">
                <div className="metric-icon-wrap blue">
                  <ShieldCheck size={24} />
                </div>
                <div className="metric-content">
                  <span className="metric-label">Avg Confidence</span>
                  <div className="metric-value-row">
                    <span className="metric-number">{results.average_confidence}%</span>
                  </div>
                </div>
              </div>

              {/* Total Particle Area */}
              <div className="metric-card amber-glow">
                <div className="metric-icon-wrap amber">
                  <Layers size={24} />
                </div>
                <div className="metric-content">
                  <span className="metric-label">Total Particle Area</span>
                  <div className="metric-value-row">
                    <span className="metric-number">{results.total_area_px.toLocaleString()}</span>
                    <span className="metric-unit">px²</span>
                  </div>
                </div>
              </div>

              {/* Contamination Level Badge */}
              <div className="metric-card severity-card" style={{ borderColor: results.contamination.color }}>
                <div
                  className="metric-icon-wrap"
                  style={{ background: `${results.contamination.color}22`, color: results.contamination.color }}
                >
                  <Activity size={24} />
                </div>
                <div className="metric-content">
                  <span className="metric-label">Contamination Level</span>
                  <div className="metric-value-row">
                    <span className="severity-badge" style={{ color: results.contamination.color }}>
                      {results.contamination.level}
                    </span>
                  </div>
                  <span className="severity-desc">{results.contamination.desc}</span>
                </div>
              </div>
            </div>

            {/* Split View: Image Viewer & Particle Details */}
            <div className="display-grid">
              {/* Image Inspection Card */}
              <div className={`viewer-card ${isZoomed ? 'zoomed' : ''}`}>
                <div className="viewer-controls">
                  <div className="tab-group">
                    <button
                      className={`tab-btn ${activeTab === 'annotated' ? 'active' : ''}`}
                      onClick={() => setActiveTab('annotated')}
                    >
                      Annotated Image ({results.count})
                    </button>
                    <button
                      className={`tab-btn ${activeTab === 'original' ? 'active' : ''}`}
                      onClick={() => setActiveTab('original')}
                    >
                      Original Sample
                    </button>
                  </div>

                  <button
                    className="zoom-toggle-btn"
                    onClick={() => setIsZoomed(!isZoomed)}
                    title={isZoomed ? 'Exit fullscreen' : 'Expand full image'}
                  >
                    {isZoomed ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
                  </button>
                </div>

                <div className="image-stage">
                  <img
                    src={activeTab === 'annotated' ? results.annotated_image : results.original_image}
                    alt="Water Specimen Result"
                    className="water-image"
                  />
                </div>

                <div className="image-footer">
                  <div className="inf-speed">
                    <span>⚡ GPU Inference: {results.inference_time_ms} ms</span>
                  </div>
                  <div className="viewer-actions">
                    <button className="secondary-btn" onClick={handleDownloadImage}>
                      <Download size={15} />
                      <span>Download Image</span>
                    </button>
                    <button className="primary-btn" onClick={handleDownloadReport}>
                      <FileText size={15} />
                      <span>Download Report</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Particle Breakdown Table */}
              <div className="particle-table-card">
                <div className="table-header-wrap">
                  <div className="flex items-center gap-2">
                    <Droplets size={18} className="text-cyan" />
                    <h4>Detected Particle Breakdown</h4>
                  </div>
                  <span className="count-pill">{results.particles.length} Items</span>
                </div>

                <div className="table-container">
                  {results.particles.length === 0 ? (
                    <div className="empty-particles">
                      <CheckCircle2 size={36} className="text-emerald" />
                      <h5>No microplastics identified</h5>
                      <p>Water specimen passes threshold purity standards.</p>
                    </div>
                  ) : (
                    <table className="particle-table">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Confidence</th>
                          <th>Area</th>
                          <th>Type Classification</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.particles.map((p) => {
                          const typeClass =
                            p.area_px > 3000
                              ? 'Large Fragment'
                              : p.area_px > 800
                              ? 'Micro-Fragment'
                              : 'Micro-Fiber';
                          return (
                            <tr key={p.id}>
                              <td className="particle-id">#{p.id}</td>
                              <td>
                                <div className="conf-cell">
                                  <div className="conf-bar-bg">
                                    <div
                                      className="conf-bar-fill"
                                      style={{ width: `${p.confidence}%` }}
                                    ></div>
                                  </div>
                                  <span>{p.confidence}%</span>
                                </div>
                              </td>
                              <td className="area-cell">{p.area_px.toLocaleString()} px²</td>
                              <td>
                                <span className={`type-tag ${typeClass.toLowerCase().replace(' ', '-')}`}>
                                  {typeClass}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                </div>

                <div className="card-footer-report">
                  <button className="report-action-btn" onClick={handleDownloadReport}>
                    <FileText size={18} />
                    <span>Download Full Audit Report (.txt)</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
