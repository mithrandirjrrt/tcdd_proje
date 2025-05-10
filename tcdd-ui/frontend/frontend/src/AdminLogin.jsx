import { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "./assets/Tcdd_logo.png";

function AdminLogin() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleLogin = () => {
    if (username === "admin123" && password === "admin123") {
      navigate("/admin");
    } else {
      setError("❌ Kullanıcı adı veya şifre yanlış.");
    }
  };

  return (
<div style={{
  background: "#e7f1ff",
  padding: "40px 20px",
  minHeight: "10vh",
  display: "flex",
  justifyContent: "center",
  alignItems: "center"
}}>



      <div style={{
        background: "white",
        padding: 30,
        borderRadius: 12,
        boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
        width: "100%",
        maxWidth: 400
      }}>
        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <img src={logo} alt="TCDD Logo" style={{ width: 150, marginBottom: 10 }} />
          <h2 style={{ color: "#003366", fontSize: 20, fontWeight: 600 }}> Admin Giriş</h2>
        </div>

        <input
          type="text"
          placeholder="Kullanıcı Adı"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          style={{
            width: "95%", marginBottom: 12, padding: 10,
            borderRadius: 6, border: "1px solid #ccc", fontSize: 14
          }}
        />
        <input
          type="password"
          placeholder="Şifre"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{
            width: "95%", marginBottom: 12, padding: 10,
            borderRadius: 6, border: "1px solid #ccc", fontSize: 14
          }}
        />

        {error && (
          <div style={{
            background: "#fdecea",
            color: "#a94442",
            padding: 10,
            borderRadius: 6,
            fontSize: 14,
            marginBottom: 12
          }}>
            {error}
          </div>
        )}

        <button
          onClick={handleLogin}
          style={{
            width: "100%",
            padding: 12,
            background: "#003366",
            color: "white",
            borderRadius: 6,
            border: "none",
            marginBottom: 10,
            fontWeight: "bold",
            cursor: "pointer"
          }}
          onMouseOver={(e) => e.currentTarget.style.background = "#002244"}
          onMouseOut={(e) => e.currentTarget.style.background = "#003366"}
        >
          Giriş Yap
        </button>

        <button
          onClick={() => navigate("/")}
          style={{
            width: "100%",
            padding: 12,
            background: "#ccc",
            borderRadius: 6,
            border: "none",
            fontWeight: "bold",
            cursor: "pointer"
          }}
          onMouseOver={(e) => e.currentTarget.style.background = "#bbb"}
          onMouseOut={(e) => e.currentTarget.style.background = "#ccc"}
        >
           Geri Dön
        </button>
      </div>
    </div>
  );
}

export default AdminLogin;
