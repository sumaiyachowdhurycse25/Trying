const express = require('express');
const app = express();
const port = 3000;
const path = require('path');

// Serve static files from the 'public' directory
// //app.use(express.static('pages'));


app.use(express.static(path.join(__dirname, 'pages')));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'pages', 'home.html'));
});

app.get('/about', (req, res) => {
  res.sendFile(path.join(__dirname, 'pages', 'about.html'));
});

app.listen(port, () => {
console.log(`Server running at http://localhost:${port}`);
});
