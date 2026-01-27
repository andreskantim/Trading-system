"""
Email Broadcaster - Envía señales de trading por correo
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
import pandas as pd
import json

project_root = Path(__file__).resolve().parents[2]


class EmailBroadcaster:
    """Envía señales por email y guarda log de envíos"""

    SENT_DIR = Path(__file__).parent / "sent"

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = None,
        sender_password: str = None,
        recipient_email: str = "dummy@example.com"  # CAMBIAR
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.SENT_DIR.mkdir(parents=True, exist_ok=True)

    def format_signals_html(self, signals: list) -> str:
        """Formatea señales como HTML para email"""
        if not signals:
            return "<p>No hay señales nuevas.</p>"

        html = "<table border='1' cellpadding='5' style='border-collapse:collapse;'>"
        html += "<tr style='background:#333;color:white;'>"
        html += "<th>Timestamp</th><th>Symbol</th><th>Señal</th><th>Precio</th><th>Estrategia</th>"
        html += "</tr>"

        for sig in signals:
            color = "#d4edda" if sig['signal_type'] == 'entry_long' else (
                "#f8d7da" if sig['signal_type'] == 'entry_short' else "#fff3cd"
            )
            html += f"<tr style='background:{color};'>"
            html += f"<td>{sig['timestamp']}</td>"
            html += f"<td><b>{sig['symbol']}</b></td>"
            html += f"<td>{sig['signal_type']}</td>"
            html += f"<td>{sig['price']:.2f}</td>"
            html += f"<td>{sig['strategy']}</td>"
            html += "</tr>"

        html += "</table>"
        return html

    def format_signals_text(self, signals: list) -> str:
        """Formatea señales como texto plano"""
        if not signals:
            return "No hay señales nuevas."

        lines = ["NUEVAS SEÑALES DE TRADING", "=" * 40, ""]
        for sig in signals:
            lines.append(f"[{sig['signal_type'].upper()}] {sig['symbol']}")
            lines.append(f"  Precio: {sig['price']:.2f}")
            lines.append(f"  Estrategia: {sig['strategy']}")
            lines.append(f"  Timestamp: {sig['timestamp']}")
            lines.append("")

        return "\n".join(lines)

    def send_email(self, signals: list, subject: str = None) -> dict:
        """
        Envía email con señales

        Args:
            signals: Lista de señales a enviar
            subject: Asunto del email (opcional)

        Returns:
            Dict con resultado del envío
        """
        if not signals:
            return {'success': True, 'sent': False, 'reason': 'no_signals'}

        if not self.sender_email or not self.sender_password:
            return self._save_pending(signals, "credentials_not_configured")

        timestamp = datetime.now()
        if subject is None:
            subject = f"[TRADING] {len(signals)} nuevas señales - {timestamp.strftime('%Y-%m-%d %H:%M')}"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email

        text_content = self.format_signals_text(signals)
        html_content = f"""
        <html>
        <body>
        <h2>Nuevas Señales de Trading</h2>
        <p>Generadas: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        {self.format_signals_html(signals)}
        <hr>
        <p style='color:gray;font-size:12px;'>Generado por Screening System</p>
        </body>
        </html>
        """

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())

            self._save_sent_log(signals, timestamp, success=True)
            return {'success': True, 'sent': True, 'count': len(signals)}

        except Exception as e:
            self._save_sent_log(signals, timestamp, success=False, error=str(e))
            return {'success': False, 'error': str(e)}

    def _save_pending(self, signals: list, reason: str) -> dict:
        """Guarda señales pendientes de envío"""
        timestamp = datetime.now()
        self._save_sent_log(signals, timestamp, success=False, error=reason)
        return {'success': True, 'sent': False, 'reason': reason, 'saved': True}

    def _save_sent_log(self, signals: list, timestamp: datetime, success: bool, error: str = None):
        """Guarda log de envío en sent/"""
        log_file = self.SENT_DIR / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        log_data = {
            'timestamp': timestamp.isoformat(),
            'recipient': self.recipient_email,
            'success': success,
            'signals_count': len(signals),
            'signals': signals,
            'error': error
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

    def broadcast_new_signals(self, signal_results: dict) -> dict:
        """
        Procesa resultados del detector y envía señales nuevas

        Args:
            signal_results: Dict con resultados por estrategia del SignalDetector

        Returns:
            Dict con resultados de broadcast por estrategia
        """
        all_new_signals = []

        for strategy_name, results in signal_results.items():
            for r in results:
                if r.get('success') and r.get('new_signals', 0) > 0:
                    signals_csv = project_root / "screening" / "signals" / strategy_name / "signals.csv"
                    if signals_csv.exists():
                        df = pd.read_csv(signals_csv, parse_dates=['timestamp'])
                        recent = df[df['symbol'] == r['symbol']].tail(r['new_signals'])
                        for _, row in recent.iterrows():
                            all_new_signals.append(row.to_dict())

        if all_new_signals:
            return self.send_email(all_new_signals)

        return {'success': True, 'sent': False, 'reason': 'no_new_signals'}
