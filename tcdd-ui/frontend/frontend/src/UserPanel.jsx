import { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "./assets/Tcdd_logo.png";
import axios from "axios";

const vagonTipleri = [
  "Fals (665 0 331/2708)", "Ks (330 1 001/2650)", "Sgss (456 8 923/9772)", "Es (552 0 002/1902)",
  "Falns (644/664 1 001/531)", "Hbbillnss (246 1 001/999)", "Ks (330 2 652/3252)", "Rilnss (354 6 001/476)",
  "Rgns (TSI) (381 6 001/500)", "Falns (664 1 532/891)", "Els (513 3 005/650)", "Sgss (456 8 351/580)",
  "Talns (TSI) (066 5 001/300)", "Habiss (285 1 001/402)", "Sgss (456 8 581/922)", "Fas (637 7 001/330)",
  "Sgs (454 0 001/452)", "Habis (275 2 001/390)", "Falns (664 2 001/300)", "Rilnss (354 6 477/524)",
  "Sgss (456 8 000/350)", "Eanoss (TSI) (537 9 192/80066)", "Rgns (TSI) (381 6 501/752)", "Ss (470 0 001/501)",
  "Lgs (442 5 001/308)", "Eanoss (537 9 001/191)", "Ss (470 1 001/200)", "Zans (783 6 001/200)",
  "Gabs (181 0 001/101)", "Zacens (TSI) (783 4 001/200)", "Uadgs (932 9 001/300)", "Zaes (795 2 001/100)",
  "Zaes (796 8 001/100)", "Gbs (1510 001/2999)", "Ks (330 0 001/999)", "Tadns (083 8 001/050)",
  "Uagoos (935 8 001/033)", "Gbs (151 3 001/501)", "Zas (795 0 050/115)", "Zaes (787 8 001/145)",
  "Zas (784 9 002/095)", "Fad (686 0 004/4008)", "Fabls (695 0 001/040)", "Gbs (1500 001/506)"
];

const komponentler = [
  "Tekerleğin Bandaj Kısmı", "Fren Pnömatik Kısım", "Sabo", "Boden", "Kapı ve Sürme Duvar",
  "Dikme (Platform Vagon)", "Yan veya Alın Duvar (Açık Vagon)", "Fren Mekanik Kısım", "Duvar",
  "Yarı Otomatik Koşum Takımı", "Yan Duvar Kapağı (Platform Vagon)",
  "Basamak/Tutamak/Merdiven/Geçit/Korkuluk/Yazı Levhaları vb. Değişik Parçalar",
  "Yaprak Susta", "Boji Yan Yastık ve Sustası", "Branda Kilitleme Tertibatı (Rils vb)",
  "Çatı ve Su Sızdırmazlığı (Kapalı Vagon)", "Dikme Desteği (Platform Vagon)", "Y 25 Bojinin Süspansiyon Sistemi",
  "Topraklama Kablosu", "Süspansiyon Bağlantıları", "El Freni", "Taban", "Dingil Kutusu",
  "Vagon Gövdesi İç Donanımı", "Yükün Dağılımı", "Kapama Tertibatı/Tespit Sportu (Kapalı Vagon)",
  "ACTS Konteyner Vagonu (Özel Tertibatlı Vagon)", "Menteşe/Pim/Sabitleme Civatası (Platform Vagon)",
  "Havalandırma Kapağı (Kapalı Vagon)", "Helezon Susta", "Dingilli Vagonlarda Süspansiyon Sportu",
  "Monoblok Tekerlek", "Alın Kapakların Kapatılması ve Çalıştırılması Tertibatı (Açık Vagon)",
  "Boşi Şasisi", "Tampon Plakası", "Vagon Şasesi", "Vagon Üzerindeki Yazı ve İşaretler",
  "Konteyner Vagonları Üzerindeki Yük Ünitelerinin Emniyete Alınmasına Yönelik Tertibat",
  "Vagon Gövdesi İskeleti", "İşletme Bozuklukları", "Payanda, Kalas, Gerdirme, Bağlantı Tertibatı",
  "Özellikle Yatay veya Düşey Aktarım için Kullanılan Özel Ekipman", "Doğrudan veya Dolaylı Bağlantı",
  "Dingil", "Bandajlı Tekerlek", "Dingil Çatalı Bağlantı Pernosu", "Dingil Çatalı Aşınma Plakası",
  "Vagon Duvarı veya Kenarı", "Tekerlek Gövdesi", "Y Bojide Manganlı Aşınma Plakası"
];

const API_BASE = import.meta.env.VITE_API_BASE;

function UserPanel() {
  const [vagonNo, setVagonNo] = useState("");
  const [vagonTipi, setVagonTipi] = useState("");
  const [komponent, setKomponent] = useState("");
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleNext = async () => {
    if (!/^[0-9]{11}$/.test(vagonNo)) {
      setError("Vagon numarası 11 haneli olmalıdır.");
      return;
    }
    if (!vagonTipi || !komponent) {
      alert("Tüm alanları doldurun.");
      return;
    }

    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        vagon_no: vagonNo,
        vagon_tipi: vagonTipi,
        komponent: komponent
      });
      navigate("/predict", { state: res.data });
    } catch (err) {
      console.error("Tahmin yapılamadı:", err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError("Tahmin yapılamadı Vagon Aktif Bakımda. Lütfen tekrar deneyin.");
      }
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", padding: 20, background: "#fff", borderRadius: 12, boxShadow: "0 0 10px rgba(0,0,0,0.05)" }}>
      <div style={{ textAlign: "center", marginBottom: 20 }}>
        <img src={logo} alt="TCDD Logo" style={{ width: 200, marginBottom: 10 }} />
        <h2 style={{ color: "#003366", fontSize: 22, marginBottom: 5 }}>Vagon Arıza Tahmin Sistemi</h2>
        <p style={{ fontSize: 14, color: "#555" }}>Lütfen aşağıdaki bilgileri eksiksiz doldurun.</p>
      </div>

      <input
        value={vagonNo}
        onChange={(e) => setVagonNo(e.target.value)}
        placeholder="Vagon No (11 hane)"
        style={{
          width: "96%", marginBottom: 15, padding: 10,
          border: "1px solid #ccc", borderRadius: 8, fontSize: 14
        }}
      />

      <select
        value={vagonTipi}
        onChange={(e) => setVagonTipi(e.target.value)}
        style={{
          width: "100%", marginBottom: 15, padding: 10,
          border: "1px solid #ccc", borderRadius: 8, fontSize: 14
        }}
      >
        <option value="">Vagon Tipi Seç</option>
        {vagonTipleri.map((v, i) => (
          <option key={i} value={v}>{v}</option>
        ))}
      </select>

      <select
        value={komponent}
        onChange={(e) => setKomponent(e.target.value)}
        style={{
          width: "100%", marginBottom: 20, padding: 10,
          border: "1px solid #ccc", borderRadius: 8, fontSize: 14
        }}
      >
        <option value="">Komponent Seç</option>
        {komponentler.map((k, i) => (
          <option key={i} value={k}>{k}</option>
        ))}
      </select>

      {error && (
        <div style={{
          background: "#ffebee",
          border: "1px solid #ffcdd2",
          borderRadius: 8,
          padding: "16px",
          marginBottom: "20px",
          color: "#b00020",
          display: "flex",
          alignItems: "center",
          gap: "10px"
        }}>
          <span style={{ fontSize: "20px" }}>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      <button
        onClick={handleNext}
        style={{
          width: "100%", padding: 12, background: "#003366",
          color: "white", fontWeight: "bold", border: "none", borderRadius: 8,
          cursor: "pointer", transition: "background 0.3s"
        }}
        onMouseOver={(e) => e.currentTarget.style.background = "#002244"}
        onMouseOut={(e) => e.currentTarget.style.background = "#003366"}
      >
        Tahmini Görüntüle
      </button>
    </div>
  );
}

export default UserPanel;
