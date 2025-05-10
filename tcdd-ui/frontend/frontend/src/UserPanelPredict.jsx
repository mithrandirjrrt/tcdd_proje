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
    const fetchActive = async () => {
      try {
        const res = await axios.get(`${API_BASE}/active_repairs`);
        setTumBakimlar(res.data || {});
      } catch {
        console.error("Aktif bakımlar alınamadı.");
      }
    };
    fetchActive();
  }, []);

  const handleActivate = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/activate_repair`, location.state);
      setResponse({ ...location.state, ...res.data });
      setActivated(true);
      const updated = await axios.get(`${API_BASE}/active_repairs`);
      setTumBakimlar(updated.data || {});
    } catch {
      alert("Bakım aktifleştirilemedi.");
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
              <div style={{ fontSize: 18, fontWeight: 600, marginTop: 4 }}>{response.prediction}</div>
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
              <h4 style={{ fontSize: 16, fontWeight: 600, marginBottom: 10, color: "#003366" }}>📊 İstasyon Kapasiteleri</h4>
              {Object.entries(capacity).map(([key, val]) => (
                <div key={key} style={{ marginBottom: 6 }}>
                  <strong>{key}</strong>: {val} / 5
                </div>
              ))}
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
              <h4 style={{ fontSize: 16, fontWeight: 600, marginBottom: 10, color: "#003366" }}>🛠️ {response.prediction} İstasyonundaki Bakımlar</h4>
              {stationRepairs.length > 0 ? (
                stationRepairs.map((rep, i) => (
                  <div key={i} style={{ marginBottom: 6, fontSize: 14 }}>
                    🔹 <strong>{rep.vagon_no}</strong> – {rep.vagon_tipi} – {rep.komponent} ({rep.neden})
                  </div>
                ))
              ) : (
                <p style={{ fontSize: 14, color: "#888" }}>Bu istasyonda aktif bakım yok.</p>
              )}

              {/* Ek olarak: tüm istasyonlar */}
              <div style={{ marginTop: 20 }}>
                <h4 style={{ fontSize: 16, fontWeight: 600, color: "#003366" }}>📌 Tüm İstasyonlardaki Aktif Bakım Sayısı</h4>
                <ul style={{ listStyle: "none", paddingLeft: 0 }}>
                  {Object.entries(tumBakimlar).map(([ist, list]) => (
                    <li key={ist}>🔹 <strong>{ist}</strong>: {list.length} bakım</li>
                  ))}
                </ul>
              </div>

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
