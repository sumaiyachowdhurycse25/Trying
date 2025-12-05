    const express = require('express');
    const { Pool } = require('pg');
    const app = express();
    const port = 3000;

    // PostgreSQL connection pool
    const pool = new Pool({
        user: 'postgres', // e.g., 'postgres'
        host: 'localhost',
        database: 'contact_db',
        password: 'Novely25',
        port: 5432,
    });

    // Middleware to parse JSON and URL-encoded data
    app.use(express.json());
    app.use(express.urlencoded({ extended: true }));

    // Serve static files (e.g., your HTML form)
    app.use(express.static('pages')); // Assuming your HTML is in a 'public' folder



    // Handle form submission
    app.post('/submit-contact', async (req, res) => {
        const { name, email, number, plan, address, message } = req.body;

        if (!name || !email || !number || !plan || !address || !message) {
            return res.status(400).send('Please fill in all required fields.');
        }

        try {
            const result = await pool.query(
                'INSERT INTO contact_form (name, email, number, plan, address, message) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *',
                [name, email, number, plan, address, message]
            );
            console.log('Message saved:', result.rows[0]);
            res.status(200).send('Message sent successfully!');
        } catch (error) {
            console.error('Error saving message:', error);
            res.status(500).send('Error sending message.');
        }
    });

    // Start the server
    app.listen(port, () => {
        console.log(`Server running at http://localhost:${port}`);
    });
