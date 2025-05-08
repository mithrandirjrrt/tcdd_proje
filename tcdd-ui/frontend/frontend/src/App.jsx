import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import UserPanel from "./UserPanel";
import AdminLogin from "./AdminLogin";
import AdminStationPanel from "./AdminStationPanel";
import UserPanelPredict from "./UserPanelPredict";

function App() {
  return (
    <Router>
      <div style={{ background: "#dbefff", minHeight: "100vh", padding: 30 }}>
        <div style={{ maxWidth: 800, margin: "auto", background: "#fff", padding: 20, borderRadius: 10, boxShadow: "0 0 10px #ccc" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
            <img src="/tcdd-logo.png" alt="TCDD Logo" style={{ height: 60 }} />
            <Link to="/admin-login">
              <button style={{ background: "#003366", color: "white", padding: "10px 16px", borderRadius: 6 }}>Admin Giriş</button>
            </Link>
          </div>

          <Routes>
            <Route path="/" element={<UserPanel />} />
            <Route path="/predict" element={<UserPanelPredict />} />
            <Route path="/admin-login" element={<AdminLogin />} />
            <Route path="/admin" element={<AdminStationPanel />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
