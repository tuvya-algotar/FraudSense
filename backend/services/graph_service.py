import json
from datetime import datetime

class GraphService:
    def __init__(self, db):
        self.db = db

    # =========================================================
    # Chain Engine Logic
    # =========================================================
    def _get_chain_record(self, user_id):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM chain_states WHERE user_id = ?", (user_id,))
        return cursor.fetchone()

    def process_chain_event(self, user_id, event):
        rec = self._get_chain_record(user_id)
        if not rec:
            self.db.execute("INSERT INTO chain_states (user_id, state, event_log, suspicion_score, last_event_time) VALUES (?, ?, ?, ?, ?)",
                            (user_id, "CLEAN", "[]", 0, datetime.now().isoformat()))
            self.db.commit()
            rec = self._get_chain_record(user_id)
        
        event_log = json.loads(rec["event_log"] or "[]")
        event_log.append({"event": event, "time": datetime.now().isoformat()})
        
        # Keep log trim to prevent unchecked growth
        if len(event_log) > 10:
            event_log = event_log[-10:]

        new_state = rec["state"]
        if event == "LOGIN_NEW_DEVICE":
            new_state = "WATCH"
        elif event == "ANALYST_CONFIRM_FRAUD":
            new_state = "BLOCKED"
        elif event == "TRANSACTION_ATTEMPT" and new_state != "BLOCKED":
            new_state = "MFA_REQUIRED"
        elif event == "FAILED_MFA":
            new_state = "BLOCKED"
        elif event == "MFA_SUCCESS" and new_state != "BLOCKED":
            new_state = "CLEAN"
            
        self.db.execute("UPDATE chain_states SET state = ?, event_log = ?, last_event_time = ? WHERE user_id = ?",
                        (new_state, json.dumps(event_log), datetime.now().isoformat(), user_id))
        self.db.commit()
        return new_state

    def get_chain_risk_boost(self, user_id):
        rec = self._get_chain_record(user_id)
        if not rec:
            return 0.0
            
        base_boost = 0.0
        if rec["state"] == "WATCH":
            base_boost = 15.0
        elif rec["state"] == "MFA_REQUIRED":
            base_boost = 25.0
        elif rec["state"] == "BLOCKED":
            base_boost = 100.0

        # Repeated suspicious pattern adds a small risk boost
        event_log = json.loads(rec["event_log"] or "[]")
        suspicious_count = sum(1 for e in event_log if e["event"] in ["LOGIN_NEW_DEVICE", "FAILED_MFA"])
            
        if suspicious_count >= 2:
            return min(100.0, base_boost + 10.0)
            
        return base_boost

    def get_current_chain_state(self, user_id):
        rec = self._get_chain_record(user_id)
        if not rec:
            return "CLEAN"
        return rec["state"]

    def reset_chain(self, user_id):
        self.db.execute("UPDATE chain_states SET state = 'CLEAN', event_log = '[]', suspicion_score = 0 WHERE user_id = ?", (user_id,))
        self.db.commit()

    # =========================================================
    # Simple Link Detection (Replaces NetworkX Graph)
    # =========================================================
    def detect_suspicious_clusters(self, current_user_id, current_merchant_id):
        cursor = self.db.cursor()
        graph_risk = 0.0
        reasons = []

        # ── 1. Device Linking ───────────────────────────────────────────────
        cursor.execute("SELECT device_id FROM user_devices WHERE user_id = ? ORDER BY last_seen DESC LIMIT 1", (current_user_id,))
        dev_row = cursor.fetchone()
        
        if dev_row and dev_row['device_id']:
            device_id = dev_row['device_id']
            cursor.execute("SELECT DISTINCT user_id FROM user_devices WHERE device_id = ? AND user_id != ?", (device_id, current_user_id))
            linked_users = [row['user_id'] for row in cursor.fetchall()]
            
            if linked_users:
                # Check if any linked user had fraud
                placeholders = ','.join('?' * len(linked_users))
                cursor.execute(f"SELECT state FROM chain_states WHERE user_id IN ({placeholders}) AND state IN ('BLOCKED', 'MFA_REQUIRED')", linked_users)
                fraud_linked = cursor.fetchall()
                
                if fraud_linked:
                    graph_risk = max(graph_risk, 75.0)  # High risk inline with 60-80 threshold
                    reasons.append("Network Risk: Device linked to previously blocked/flagged users")
                elif len(linked_users) >= 2:
                    graph_risk = max(graph_risk, 30.0)
                    reasons.append(f"Network Risk: Device shared by {len(linked_users)} other users")

        # ── 2. Merchant Linking ─────────────────────────────────────────────
        if current_merchant_id:
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as user_count 
                FROM transactions 
                WHERE merchant_id = ? AND timestamp > datetime('now', '-24 hours')
            """, (current_merchant_id,))
            merch_row = cursor.fetchone()
            
            if merch_row and merch_row['user_count'] >= 5:
                graph_risk = max(graph_risk, 25.0)
                reasons.append(f"Coordinated Activity: {merch_row['user_count']} users targeting same merchant")

        return {
            "graph_risk": min(100.0, graph_risk),
            "reasons": reasons
        }
