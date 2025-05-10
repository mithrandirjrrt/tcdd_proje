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
  const [error, setError] = useState(false);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        // Eğer state gelmemişse test verisi kullan
        const inputData = location.state || {
          vagon_no: "12345678901",
          vagon_tipi: "Falns",
          komponent: "Fren Pnömatik Kısım"
        };

        console.log("📦 Gönderilen tahmin verisi:", inputData);

        const res = await axios.post(`${API_BASE}/predict`, inputData);

        console.log("✅ API yanıtı:", res.data);

        if (!res.data || !res.data.prediction) {
          setError(true);
          return;
        }

        setResponse(res.data);
        setCapacity(res.data.capacity_map || {});
        setStationRepairs(res.data.station_repairs || []);
      } catch (err) {
        console.error("❌ API hatası:", err);
        setError(true);
      }
    };

    fetchPrediction();
  }, [location]);

  if (error) {
    return (
      <div>
        <h3 style={{ color: "red" }}>❌ Tahmin alınamadı. Lütfen bilgileri kontrol edin.</h3>
        <button onClick={() => navigate("/")} style={{ marginTop: 20 }}>⬅️ Geri Dön</button>
      </div>
    );
  }

  if (!response) return <div>⏳ Yükleniyor...</div>;

  return (
    <div>
      <h2 style={{ color: "#003366" }}>Tahmin Sonuçları:</h2>

      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 20 }}>
        <div style={{ background: "#e6f2ff", padding: 10, flex: 1, marginRight: 10 }}>
          <strong>📍 Yönlendirilen İstasyon:</strong><br />
          {response.prediction}
        </div>
        <div style={{ background: "#fff3cd", padding: 10, flex: 1 }}>
          <strong>🔧 Tahmini Tamire Tutulma Nedeni:</strong><br />
          {response.neden || "N/A"}
        </div>
      </div>

      {response.replaced && (
        <div style={{ marginBottom: 20, color: "red" }}>
          ❗ Asıl önerilen istasyon: <strong>{response.fallback}</strong> (kapasite yetersizliği nedeniyle atanamadı)
        </div>
      )}

      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ flex: 1, maxHeight: 250, overflowY: "auto", border: "1px solid #ccc", padding: 10 }}>
          <h4>📊 Tüm İstasyonların Kapasiteleri</h4>
          {Object.entries(capacity).map(([key, val]) => (
            <div key={key}><strong>{key}</strong>: {val} / 5</div>
          ))}
        </div>

        <div style={{ flex: 1, maxHeight: 250, overflowY: "auto", border: "1px solid #ccc", padding: 10 }}>
          <h4>🛠️ {response.prediction} İstasyonundaki Mevcut Bakımlar</h4>
          {stationRepairs.length > 0 ? stationRepairs.map((rep, i) => (
            <div key={i} style={{ marginBottom: 5 }}>
              🔹 <strong>{rep.vagon_no}</strong> - {rep.vagon_tipi} - {rep.komponent} ({rep.neden})
            </div>
          )) : <p>İşlemde bakım yok.</p>}
        </div>
      </div>

      <button
        onClick={() => navigate("/")}
        style={{ marginTop: 20, padding: 10, background: "#ddd" }}
      >
        ⬅️ Geri Dön
      </button>
    </div>
  );
}

export default PredictPage;
