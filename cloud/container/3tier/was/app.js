const express = require("express");
const mariadb = require("mysql2");
d;
const app = express();
const db = mariadb.createConnection({
  host: "mariadb-3tier",
  user: "user",
  password: "password123",
  database: "skala",
});
app.get("/users", (req, res) => {
  db.query("SELECT * FROM users", (err, rows) => {
    if (err) return res.status(500).json({ error: err });
    res.json(rows);
  });
});
app.listen(8080, () => console.log("WAS listening on 8080"));
