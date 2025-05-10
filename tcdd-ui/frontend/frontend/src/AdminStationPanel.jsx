import { useState, useEffect } from "react";
import axios from "axios";
import Select from "react-select";
import logo from "./assets/Tcdd_logo.png"
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

  useEffect(() => {
    const saved = localStorage.getItem("bakimlar");
    if (saved) {
      setBakimlar(JSON.parse(saved));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("bakimlar", JSON.stringify(bakimlar));
  }, [bakimlar]);
  useEffect(() => {
    axios.get("http://127.0.0.1:8000/active_repairs")
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

  const handleBitti = async (istasyon, index) => {
    const secili = bakimlar[istasyon][index];
    if (!secili.neden) {
      alert("Lütfen tamir nedenini seçin!");
      return;
    }
    try {
      await axios.post("http://127.0.0.1:8000/complete_repair", {
        vagon_no: secili.vagon_no,
        vagon_tipi: secili.vagon_tipi,
        komponent: secili.komponent,
        neden: secili.neden,
        istasyon: istasyon
      });
    } catch (err) {
      alert("API'ye gönderilemedi ❌");
      return;
    }

    const guncel = { ...bakimlar };
    guncel[istasyon].splice(index, 1);
    if (guncel[istasyon].length === 0) delete guncel[istasyon];
    setBakimlar(guncel);
  };

  return (
    <div style={{ textAlign:"center"}}>
       <img src={logo} alt="TCDD Logo" style={{ width: 100, marginBottom: 10 }} />
      <h2 style={{ color: "#003366" }}>🛠️ Bakım İstasyonları</h2>

      <Select
        options={istasyonlar.map(i => ({ value: i, label: i }))}
        placeholder="İstasyon Ara ve Seçin..."
        onChange={(selected) => setSeciliIstasyon(selected?.value || "")}
        isClearable
      />

      {seciliIstasyon && bakimlar[seciliIstasyon]?.length > 0 ? (
        <table border="1" width="100%" cellPadding="6" style={{ marginTop: 20 }}>
          <thead>
            <tr>
              <th>Vagon No</th>
              <th>Vagon Tipi</th>
              <th>Komponent</th>
              <th>Tamir Nedeni</th>
              <th>İşlem</th>
            </tr>
          </thead>
          <tbody>
            {bakimlar[seciliIstasyon].map((item, idx) => (
              <tr key={idx}>
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
                  />
                </td>
                <td>
                  <button onClick={() => handleBitti(seciliIstasyon, idx)}>✅ Bitti</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p style={{ marginTop: 20 }}>{seciliIstasyon ? "Bu istasyonda aktif bakım yok." : "Lütfen bir istasyon seçin."}</p>
      )}

      <button
        style={{ marginTop: 30, background: "#ccc", padding: 10 }}
        onClick={() => window.location.href = "/"}
      >
        ⬅️ Geri Dön
      </button>
    </div>
  );
}

export default AdminStationPanel;
