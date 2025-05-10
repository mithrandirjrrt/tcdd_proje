import { useState, useEffect } from "react";
import axios from "axios";
import Select from "react-select";
import logo from "./assets/Tcdd_logo.png";

const API_BASE = import.meta.env.VITE_API_BASE;

const istasyonlar = [
  "Adana", "Afyon", "Alsancak", "Ankara", "Arifiye", "Balıkesir", "Bandırma", "Bilecik", "Biçerova",
  "Bostankaya", "Burdur", "Demirdağ", "Denizli", "Derince", "Dinar", "Divriği", "Diyarbakır", "Elazığ",
  "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Gebze/Haydarpaşa", "Halkalı", "Hekimhan", "Irmak",
  "Kapıkule", "Kapıköy", "Karabük", "Kayseri", "Konya", "Kurtalan", "Köseköy", "Kütahya", "Malatya",
  "Mersin", "Sivas", "Soma", "Tatvan", "Tavşanlı", "Uşak", "Van", "Yakacık", "Zonguldak", "Çankırı",
  "Çatalağzı", "Ülkü", "İskenderun"
].sort();

const nedenler = [
  "ACTS Konteyner Vagonu (Özel Tertibatlı Vagon)",
  "Alın Kapakların Kapatılması ve Çalıştırılması Tertibatı (Açık Vagon)",
  "Bandajlı Tekerlek", "Basamak/Tutamak/Merdiven/Geçit/Korkuluk/Yazı Levhaları vb. Değişik Parçalar",
  "Boden", "Boji Yan Yastık ve Sustası", "Boşi Şasisi", "Branda Kilitleme Tertibatı (Rils vb)",
  "Dikme", "Dikme (Platform Vagon)", "Dikme Desteği (Platform Vagon)", "Dingil",
  "Dingil Kutusu", "Dingil Çatalı Aşınma Plakası", "Dingil Çatalı Bağlantı Pernosu",
  "Dingilli Vagonlarda Süspansiyon Sportu", "Doğrudan veya Dolaylı Bağlantı", "Duvar",
  "El Freni", "Fren Mekanik Kısım", "Fren Pnömatik Kısım", "Havalandırma Kapağı (Kapalı Vagon)",
  "Helezon Susta", "Kapama Tertibatı/Tespit Sportu (Kapalı Vagon)", "Kapı ve Sürme Duvar",
  "Konteyner Vagonları Üzerindeki Yük Ünitelerinin Emniyete Alınmasına Yönelik Tertibat",
  "Menteşe/Pim/Sabitleme Civatası (Platform Vagon)", "Monoblok Tekerlek", "Paketleme, Yük Bağlama",
  "Payanda, Kalas, Gerdirme, Bağlantı Tertibatı", "Sabo", "Süspansiyon Bağlantıları", "Taban",
  "Tampon Plakası", "Tekerlek Gövdesi", "Tekerleğin Bandaj Kısmı", "Topraklama Kablosu",
  "Vagon Duvarı veya Kenarı", "Vagon Gövdesi İskeleti", "Vagon Gövdesi İç Donanımı",
  "Vagon Üzerindeki Yazı ve İşaretler", "Vagon Şasesi", "Y 25 Bojinin Süspansiyon Sistemi",
  "Y Bojide Manganlı Aşınma Plakası", "Yan Duvar Kapağı (Platform Vagon)",
  "Yan veya Alın Duvar (Açık Vagon)", "Yaprak Susta", "Yarı Otomatik Koşum Takımı",
  "Yükün Dağılımı", "Çatı ve Su Sızdırmazlığı (Kapalı Vagon)",
  "Özellikle Yatay veya Düşey Aktarım için Kullanılan Özel Ekipman", "İşletme Bozuklukları"
].sort();

function AdminStationPanel() {
  const [seciliIstasyon, setSeciliIstasyon] = useState("");
  const [bakimlar, setBakimlar] = useState({});
  const [vagonNo, setVagonNo] = useState("");
  const [gecmisBakimlar, setGecmisBakimlar] = useState([]);
  const [gecmisLoading, setGecmisLoading] = useState(false);
  const [gecmisError, setGecmisError] = useState(null);
  const [lastLogin, setLastLogin] = useState("");

  useEffect(() => {
    const saved = localStorage.getItem("bakimlar");
    if (saved) setBakimlar(JSON.parse(saved));
    const loginTime = localStorage.getItem("lastLogin");
    if (loginTime) setLastLogin(loginTime);
  }, []);

  useEffect(() => {
    localStorage.setItem("bakimlar", JSON.stringify(bakimlar));
  }, [bakimlar]);

  useEffect(() => {
    axios.get(`${API_BASE}/active_repairs`)
      .then(res => {
        setBakimlar(res.data);
        localStorage.setItem("bakimlar", JSON.stringify(res.data));
      })
      .catch(() => console.error("Aktif bakımlar alınamadı ❌"));
  }, []);

  const handleNedenDegistir = (istasyon, index, yeniNeden) => {
    const guncel = { ...bakimlar };
    guncel[istasyon][index].neden = yeniNeden;
    setBakimlar(guncel);
  };

  const handleBitti = async (vagon) => {
    try {
      const now = new Date();
      const formattedDate = now.toLocaleString('tr-TR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });

      const response = await axios.post(`${API_BASE}/complete_repair`, {
        vagon_no: vagon.vagon_no,
        vagon_tipi: vagon.vagon_tipi,
        komponent: vagon.komponent,
        neden: vagon.neden,
        istasyon: seciliIstasyon,
        completion_time: formattedDate,
        activation_time: vagon.activation_time
      });

      if (response.status === 200) {
        alert("Bakım başarıyla tamamlandı!");
        // Aktif bakımları yeniden yükle
        const res = await axios.get(`${API_BASE}/active_repairs`);
        setBakimlar(res.data);
      }
    } catch (error) {
      console.error("Bakım tamamlanırken hata:", error);
      alert("Bakım tamamlanırken bir hata oluştu!");
    }
  };

  const handleGecmisAra = async () => {
    if (!vagonNo || vagonNo.length < 5) {
      alert("Geçerli bir vagon no girin");
      return;
    }
    setGecmisLoading(true);
    setGecmisError(null);
    try {
      const res = await axios.get(`${API_BASE}/history?vagon_no=${vagonNo}`);
      if (res.data && res.data.length > 0) {
        setGecmisBakimlar(res.data);
      } else {
        setGecmisError("Geçmiş bakım verisi bulunamadı.");
      }
    } catch (error) {
      console.error("Geçmiş bakım verisi alınamadı:", error);
      setGecmisError("Geçmiş bakım verisi alınamadı. Lütfen tekrar deneyin.");
    } finally {
      setGecmisLoading(false);
    }
  };

  return (
    <div style={{ background: "#e7f1ff", minHeight: "100vh", padding: "40px 20px" }}>
      <div style={{ maxWidth: 1024, margin: "0 auto", background: "white", borderRadius: 12, padding: 30, boxShadow: "0 8px 24px rgba(0,0,0,0.05)", textAlign: "center" }}>
        <img src={logo} alt="TCDD Logo" style={{ width: 100, marginBottom: 10 }} />
        <h2 style={{ color: "#003366", marginBottom: 20 }}>🛠️ Bakım İstasyonları</h2>
        {lastLogin && (
          <p style={{ color: "#666", fontSize: 14, marginBottom: 20 }}>
            Son giriş: {lastLogin}
          </p>
        )}

        <div style={{ marginBottom: 30 }}>
          <Select options={istasyonlar.map(i => ({ value: i, label: i }))} placeholder="İstasyon Ara ve Seçin..." onChange={(s) => setSeciliIstasyon(s?.value || "")} isClearable />
        </div>

        <div style={{ color: "#003366",marginBottom: 30, textAlign: "left" }}>
          <label><strong>Vagon No ile Geçmiş Bakım Ara:</strong></label>
          <div style={{ display: "flex", gap: 10, marginTop: 6 }}>
            <input type="text" placeholder="örn: 12345678901" value={vagonNo} onChange={(e) => setVagonNo(e.target.value)} style={{ flex: 1, padding: 8, borderRadius: 6, border: "1px solid #ccc" }} />
            <button onClick={handleGecmisAra} style={{ padding: "8px 16px", background: "#003366", color: "white", border: "none", borderRadius: 6 }}>Ara</button>
          </div>
        </div>

        {gecmisLoading && <p>⏳ Yükleniyor...</p>}
        {gecmisError && <p style={{ color: "red" }}>{gecmisError}</p>}

        {gecmisBakimlar.length > 0 && (
          <div style={{ marginTop: 20, background: "#f9f9f9", padding: 16, borderRadius: 8, border: "1px solid #ddd" }}>
            <h4 style={{ marginBottom: 14, color: "#003366" }}> Geçmiş Bakım Kayıtları</h4>
            <table width="100%" cellPadding="8" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ backgroundColor: "#f3f6f9", fontWeight: "bold", color: "#003366" }}>
                  <td>Vagon No</td>
                  <td>Vagon Tipi</td>
                  <td>Komponent</td>
                  <td>Neden</td>
                  <td>İstasyon</td>
                  <td>Giriş Zamanı</td>
                  <td>Çıkış Zamanı</td>
                </tr>
              </thead>
              <tbody>
                {gecmisBakimlar.map((b, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #eee" }}>
                    <td><strong>{b.vagon_no}</strong></td>
                    <td>{b.vagon_tipi}</td>
                    <td>{b.komponent}</td>
                    <td>{b.neden}</td>
                    <td>{b.istasyon}</td>
                    <td>{b.activation_time}</td>
                    <td>{b.completion_time}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {seciliIstasyon && bakimlar[seciliIstasyon]?.length > 0 ? (
          <div style={{ overflowX: "auto", borderRadius: 8, boxShadow: "0 0 10px rgba(0,0,0,0.05)", border: "1px solid #ddd", padding: 16, background: "#fefefe", marginTop: 30 }}>
            <table width="100%" cellPadding="8" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ backgroundColor: "#f3f6f9", fontWeight: "bold", color: "#003366" }}>
                  <td>Vagon No</td>
                  <td>Vagon Tipi</td>
                  <td>Komponent</td>
                  <td>Tamir Nedeni</td>
                  <td>Giriş Zamanı</td>
                  <td>İşlem</td>
                </tr>
              </thead>
              <tbody>
                {bakimlar[seciliIstasyon].map((item, idx) => (
                  <tr key={idx} style={{ borderBottom: "1px solid #eee" }}>
                    <td>{item.vagon_no}</td>
                    <td>{item.vagon_tipi}</td>
                    <td>{item.komponent}</td>
                    <td>
                      <Select
                        options={nedenler.map(n => ({ value: n, label: n }))}
                        placeholder="Neden Seçin..."
                        value={item.neden ? { value: item.neden, label: item.neden } : null}
                        onChange={(e) => handleNedenDegistir(seciliIstasyon, idx, e?.value || "")}
                        isClearable
                        menuPortalTarget={document.body}
                        menuPosition="fixed"
                        styles={{ menuPortal: base => ({ ...base, zIndex: 9999 }) }}
                      />
                    </td>
                    <td>{item.activation_time}</td>
                    <td>
                      <button onClick={() => handleBitti(item)} style={{ background: "#28a745", color: "white", padding: "6px 12px", borderRadius: 6, border: "none", cursor: "pointer" }}>
                        ✅ Bitti
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ marginTop: 20, fontStyle: "italic", color: "#666" }}>
            {seciliIstasyon ? "Bu istasyonda aktif bakım yok." : "Lütfen bir istasyon seçin."}
          </p>
        )}

        <button onClick={() => window.location.href = "/"} style={{ marginTop: 30, background: "#ccc", padding: "10px 20px", borderRadius: 6, border: "none", cursor: "pointer" }}>
          ⬅️ Geri Dön
        </button>
      </div>
    </div>
  );
}

export default AdminStationPanel;