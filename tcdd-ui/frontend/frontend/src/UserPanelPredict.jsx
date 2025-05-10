import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE;

function PredictPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [response, setResponse] = useState(null);
  const [capacity, setCapacity] = useState({});
  const [stationRepairs, setStationRepairs] = useState([]);
  const [tumBakimlar, setTumBakimlar] = useState({});
  const [error, setError] = useState(false);
  const [activated, setActivated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fallbackInfo, setFallbackInfo] = useState(null);

  useEffect(() => {
    if (!location.state) return;

    const fetchPrediction = async () => {
      try {
        const res = await axios.post(`${API_BASE}/predict`, location.state);
        setResponse(res.data);
        
        // Aktif bakımları getir
        const activeRepairs = await axios.get(`${API_BASE}/active_repairs`);
        setTumBakimlar(activeRepairs.data || {});
      } catch (err) {
        console.error("Tahmin alınamadı:", err);
        setError(err.response?.data?.detail || "Tahmin alınamadı. Lütfen tekrar deneyin.");
      }
    };

    fetchPrediction();
  }, [location.state]);

  const handleActivate = async () => {
    setLoading(true);
    try {
      const now = new Date();
      const formattedDate = now.toLocaleString('tr-TR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });

      const response = await axios.post(`${API_BASE}/activate_repair`, {
        ...location.state,
        activation_time: formattedDate
      });

      if (response.status === 200) {
        setResponse(prev => ({
          ...prev,
          prediction: response.data.assigned_station,
          preferred_station: response.data.preferred_station
        }));
        
        setActivated(true);
        const updated = await axios.get(`${API_BASE}/active_repairs`);
        setTumBakimlar(updated.data || {});
      }
    } catch (error) {
      console.error("Bakım aktivasyonu sırasında hata:", error);
      alert(error.response?.data?.message || "Bakım aktivasyonu sırasında bir hata oluştu!");
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  if (error) {
    return (
      <div style={{
        background: "#e7f1ff",
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        color: "#b00020"
      }}>
        ❌ Tahmin alınamadı. Lütfen tekrar deneyin.
      </div>
    );
  }

  if (!response) return (
    <div style={{
      background: "#e7f1ff",
      minHeight: "100vh",
      display: "flex",
      justifyContent: "center",
      alignItems: "center"
    }}>
      ⏳ Yükleniyor...
    </div>
  );

  return (
    <div style={{ background: "#e7f1ff", minHeight: "100vh", padding: "40px 20px" }}>
      <div style={{
        maxWidth: 960,
        margin: "0 auto",
        background: "white",
        borderRadius: 12,
        padding: 30,
        boxShadow: "0 8px 24px rgba(0,0,0,0.05)"
      }}>
        {/* Başlık */}
        <h2 style={{
          fontSize: 24,
          fontWeight: 600,
          color: "#003366",
          marginBottom: 24,
          textAlign: "center"
        }}>
          🔍 Tahmin Sonuçları
        </h2>

        {/* Gri blok */}
        <div style={{
          background: "#f8f9fa",
          padding: 24,
          borderRadius: 10
        }}>
          {/* Üst kutular */}
          <div style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 16,
            marginBottom: 20
          }}>
            <div style={{
              flex: 1,
              minWidth: 280,
              background: "#eaf4ff",
              padding: 16,
              borderRadius: 8,
              border: "1px solid #cce1ff"
            }}>
              <div style={{ fontSize: 14, color: "#555" }}>📍 Yönlendirilen İstasyon</div>
              <div style={{ fontSize: 18, fontWeight: 600, marginTop: 4 }}>
                {response.prediction}
                {response.preferred_station && response.preferred_station !== response.prediction && (
                  <div style={{ fontSize: 14, color: "#666", marginTop: 4 }}>
                    <span style={{ color: "#b33a3a" }}>❗ Tercih edilen istasyon ({response.preferred_station}) dolu olduğu için yönlendirildiniz.</span>
                  </div>
                )}
              </div>
            </div>

            <div style={{
              flex: 1,
              minWidth: 280,
              background: "#fff8dc",
              padding: 16,
              borderRadius: 8,
              border: "1px solid #ffe6a8"
            }}>
              <div style={{ fontSize: 14, color: "#555" }}>🔧 Tahmini Arıza Nedeni</div>
              <div style={{ fontSize: 16, fontWeight: 500, marginTop: 4 }}>{response.neden || "N/A"}</div>
            </div>
          </div>

          {/* Fallback Bilgisi */}
          {activated && response.replaced && (
            <div style={{
              marginBottom: 20,
              color: "#b33a3a",
              background: "#ffe6e6",
              padding: 10,
              border: "1px solid #ffb3b3",
              borderRadius: 8,
              textAlign: "center"
            }}>
              ❗ Asıl önerilen istasyon <strong>{response.fallback}</strong> dolu olduğu için <strong>{response.prediction}</strong> istasyonuna yönlendirildiniz.
            </div>
          )}

          {/* Aktifleştir Butonu */}
          {!activated ? (
            <button
              onClick={handleActivate}
              style={{ marginTop: 20, padding: "10px 20px", background: "#003366", color: "white", borderRadius: 6, border: "none", cursor: "pointer" }}
              disabled={loading}
            >
              {loading ? "Aktif ediliyor..." : "✅ Bakımı Aktifleştir"}
            </button>
          ) : (
            <p style={{ marginTop: 20, color: "green" }}>✅ Bakım başarıyla aktifleştirildi.</p>
          )}

          {/* Alt kutular */}
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginTop: 30 }}>
            <div style={{
              flex: 1,
              minWidth: 300,
              background: "#fff",
              border: "1px solid #ddd",
              borderRadius: 8,
              padding: 16,
              maxHeight: 250,
              overflowY: "auto"
            }}>
              <h4 style={{ fontSize: 16, fontWeight: 600, marginBottom: 10, color: "#003366" }}>📊 Tüm İstasyonlardaki Aktif Bakım Sayısı</h4>
              <ul style={{ listStyle: "none", paddingLeft: 0 }}>
                {Object.entries(tumBakimlar).map(([ist, list]) => (
                  <li key={ist}>🔹 <strong>{ist}</strong>: {list.length} / 5 bakım</li>
                ))}
                {Object.keys(tumBakimlar).length === 0 && <li>Hiç bir istasyonda bakım yoktur.</li>}
              </ul>
            </div>

            <div style={{
              flex: 1,
              minWidth: 300,
              background: "#fff",
              border: "1px solid #ddd",
              borderRadius: 8,
              padding: 16,
              maxHeight: 250,
              overflowY: "auto"
            }}>
              <h4 style={{ fontSize: 16, fontWeight: 600, marginBottom: 10, color: "#003366" }}>
                🛠️ {response.prediction} İstasyonundaki Bakımlar
                {response.preferred_station && response.preferred_station !== response.prediction && (
                  <span style={{ fontSize: 14, color: "#666", marginLeft: 8 }}>
                    (Tercih edilen {response.preferred_station} istasyonu dolu olduğu için)
                  </span>
                )}
              </h4>
              {tumBakimlar[response.prediction]?.length > 0 ? (
                <table width="100%" cellPadding="8" style={{ borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ backgroundColor: "#f3f6f9", fontWeight: "bold", color: "#003366" }}>
                      <td>Vagon No</td>
                      <td>Vagon Tipi</td>
                      <td>Komponent</td>
                      <td>Neden</td>
                      <td>Giriş Zamanı</td>
                    </tr>
                  </thead>
                  <tbody>
                    {tumBakimlar[response.prediction].map((rep, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid #eee" }}>
                        <td><strong>{rep.vagon_no}</strong></td>
                        <td>{rep.vagon_tipi}</td>
                        <td>{rep.komponent}</td>
                        <td>{rep.neden}</td>
                        <td>{rep.activation_time}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p>Henüz bakım yok.</p>
              )}
            </div>
          </div>
        </div>

        {/* Geri butonu */}
        <div style={{ marginTop: 30 }}>
          <button
            onClick={() => navigate("/")}
            style={{
              padding: "10px 20px",
              background: "#e0e0e0",
              borderRadius: 6,
              border: "none",
              cursor: "pointer",
              fontSize: 14
            }}
            onMouseOver={(e) => e.currentTarget.style.background = "#d0d0d0"}
            onMouseOut={(e) => e.currentTarget.style.background = "#e0e0e0"}
          >
            ⬅️ Geri Dön
          </button>
        </div>
      </div>
    </div>
  );
}

export default PredictPage;
