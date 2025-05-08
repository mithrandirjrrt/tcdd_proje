import { useState } from "react";
import { useNavigate } from "react-router-dom";

function AdminLogin() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleLogin = () => {
    if (username === "admin123" && password === "admin123") {
      navigate("/admin");
    } else {
      setError("Kullanıcı adı veya şifre yanlış.");
    }
  };

  return (
    <div style={{ padding: 30, maxWidth: 400, margin: "auto", background: "#fff", borderRadius: 10, boxShadow: "0 0 10px #ccc" }}>
      <h2 style={{ color: "#003366" }}>🔐 Admin Girişi</h2>
      <input
        type="text"
        placeholder="Kullanıcı Adı"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        style={{ width: "100%", marginBottom: 10, padding: 8 }}
      />
      <input
        type="password"
        placeholder="Şifre"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        style={{ width: "100%", marginBottom: 10, padding: 8 }}
      />
      {error && <p style={{ color: "red" }}>{error}</p>}
      <button
        onClick={handleLogin}
        style={{ width: "100%", padding: 10, background: "#003366", color: "white", marginBottom: 10 }}
      >
        Giriş Yap
      </button>
      <button
        onClick={() => navigate("/")}
        style={{ width: "100%", padding: 10, background: "#ccc" }}
      >
        Geri Dön
      </button>
    </div>
  );
}

export default AdminLogin;
