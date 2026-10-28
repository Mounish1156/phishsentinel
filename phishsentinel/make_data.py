"""Generate the locally authored, synthetic, defanged corpus (no real people, credentials or live URLs)."""
import random
import pandas as pd
from pathlib import Path

NAMES = ["Priya", "Marcus", "Elena", "Tomas", "Aisha", "Chen", "Noah", "Fatima", "Lucas", "Mei"]
URL = {
    "lookalike_domain": "hxxps://portal-secure-login[.]example-verify[.]test/{r}",
    "shortener": "hxxp://bit[.]ly/{r}",
    "ip_host": "hxxp://192[.]0[.]2[.]{r}/login",
    "internal_portal": "hxxps://intranet[.]company[.]test/help",
    "none": "",
}
C = {
 "P1_credential_harvest": dict(label=1, ch="email", sd="lookalike", uf="lookalike_domain", urg=0,
  sub=["Mailbox almost full", "Unusual sign-in detected", "Account verification required"],
  op=["Dear {n}, your mailbox is nearly full.", "Hello {n}, we noticed unusual sign-in activity.", "Attention {n}: your account needs verification.", "Hi {n}, a security check is required."],
  bd=["Verify your password immediately at {u} to avoid suspension.", "Confirm your login details at {u} or your account will be locked.", "Re-enter your credentials at {u} so we can restore access.", "Your account will be suspended within 24 hours unless you sign in at {u}."],
  cl=["Final notice!", "IT Security Team", "Do not ignore this message.", "Act now."]),
 "P2_vague_favour": dict(label=1, ch="email", sd="free_mail", uf="shortener", urg=0,
  sub=["Quick favour", "Sorting something out", "Small thing before the weekend"],
  op=["Hi {n}, hope your week is going well.", "Hey {n}, are you around for a minute?", "Morning {n}, one small thing.", "Hi {n}, I need a hand with something."],
  bd=["The finance side flagged a hiccup with your pay details; please tidy it up here: {u}", "I am stuck in meetings, could you sort the supplier transfer details through {u} for me?", "There is a mismatch on your record, just re-key the numbers at {u} when you can.", "Can you pick up the new payment form from {u} and send it back by {d}?"],
  cl=["Thanks, really appreciate it.", "Sent from my phone.", "Let me know once done.", "Cheers."]),
 "P3_parcel_sms": dict(label=1, ch="sms", sd="short_code", uf="shortener", urg=1,
  sub=["Parcel on hold", "Delivery fee due", "Missed delivery"],
  op=["Parcel notice:", "Your package could not be delivered.", "Courier alert for {n}:", "Delivery on hold."],
  bd=["Pay the small redelivery fee at {u}", "Confirm your address and card at {u} today", "Reschedule at {u} or it returns to sender", "Release your parcel via {u} within 12 hours"],
  cl=["Reply STOP to opt out.", "Act today.", "Thank you.", "Last attempt."]),
 "P4_invoice_giftcard": dict(label=1, ch="email", sd="lookalike", uf="ip_host", urg=1,
  sub=["Overdue invoice", "Urgent request from the director", "Gift cards needed"],
  op=["{n}, I need your help urgently.", "Dear {n}, the invoice below is overdue.", "Hello {n}, this is confidential.", "{n}, are you at your desk?"],
  bd=["Buy gift cards for the client event and send me the codes; details at {u}", "Wire the outstanding amount today using the form at {u}", "Review the attached invoice at {u} and approve payment right away", "Keep this between us and settle the balance via {u}"],
  cl=["Reply asap.", "Do not call, I am travelling.", "Regards, the Director.", "Urgent!"]),
 "P5_mfa_helpdesk": dict(label=1, ch="sms", sd="short_code", uf="lookalike_domain", urg=0,
  sub=["MFA prompt", "Helpdesk callback", "Token expired"],
  op=["Helpdesk here, {n}.", "Your sign-in token expired.", "IT notice for {n}:", "We are rolling out new sign-in rules."],
  bd=["Approve the prompt on your phone and enter the code at {u}", "Read the six digit code back to us or open {u}", "Re-register your device at {u} to keep access", "Confirm your login at {u} so the update completes"],
  cl=["Reply with the code.", "Thanks, IT.", "This expires soon.", "Do not share with others."]),
 "L1_password_reset": dict(label=0, ch="email", sd="internal", uf="internal_portal", urg=1,
  sub=["Password reset requested", "Your password reset link", "Reset confirmation"],
  op=["Hello {n}, we received a request to reset your password.", "Hi {n}, a password reset was requested for your staff account.", "Dear {n}, your password was changed today.", "{n}, here is your password reset confirmation."],
  bd=["If you did not request this, ignore this message. Visit the portal directly at {u}.", "No action is needed unless this was not you; open the help page {u} from your bookmarks.", "For your security, go to the portal directly rather than using links in email: {u}", "You can finish the reset within 30 minutes from the intranet: {u}"],
  cl=["IT Service Desk", "This is an automated message.", "Do not reply.", "Thank you."]),
 "L2_payroll_notice": dict(label=0, ch="email", sd="internal", uf="none", urg=1,
  sub=["Payroll update", "Payslip available", "Payroll cut-off reminder"],
  op=["Hi {n}, the monthly payroll runs on {d}.", "Hello {n}, your payslip is ready.", "Reminder: payroll closes {d}.", "Dear {n}, please check your payroll details."],
  bd=["Confirm any bank detail changes in the HR system by {d}; do not click links in email.", "Submit corrections through the HR portal before the deadline.", "Contact payroll by phone if anything looks wrong.", "Your payslip is in the HR system under Documents."],
  cl=["Payroll Team", "Automated notice.", "Thank you for your help.", "Regards, HR."]),
 "L3_it_maintenance": dict(label=0, ch="email", sd="internal", uf="internal_portal", urg=0,
  sub=["Planned maintenance", "System downtime this weekend", "Update schedule"],
  op=["Hello {n}, planned maintenance is scheduled for {d}.", "Hi all, the file servers restart on {d}.", "Notice: the VPN will be unavailable briefly.", "Dear {n}, updates install overnight."],
  bd=["Save your work beforehand; status page: {u}", "No action is required from you.", "Expect short interruptions to email and shared drives.", "Questions can go to the service desk."],
  cl=["IT Operations", "Thanks for your patience.", "Regards, IT.", "Have a good week."]),
 "L4_benefits_deadline": dict(label=0, ch="email", sd="internal", uf="internal_portal", urg=1,
  sub=["Benefits enrolment closes soon", "Action needed: benefits election", "Last day to update benefits"],
  op=["Hi {n}, benefits enrolment closes {d}.", "Reminder for {n}: elections are due soon.", "Dear {n}, open enrolment ends this week.", "Hello {n}, your benefits choices are pending."],
  bd=["Please make your selections in the HR portal ({u}) before the deadline.", "If you take no action your current plan continues.", "It is important to finish this by {d}.", "Log in through the intranet as usual to review options."],
  cl=["People Team", "Urgent only if you plan changes.", "Thank you.", "HR Benefits."]),
 "L5_courier_real": dict(label=0, ch="sms", sd="internal", uf="none", urg=0,
  sub=["Delivery update", "Parcel received at reception", "Courier drop-off"],
  op=["Reception: a parcel for {n} arrived.", "Your delivery was left at the front desk.", "Hi {n}, your order was received.", "Mailroom notice for {n}."],
  bd=["Collect it any time before {d} with your staff card.", "No payment is needed.", "It is on the shelf by the mailroom.", "Ask the front desk if you cannot find it."],
  cl=["Facilities", "Thanks.", "Mailroom team.", "See you soon."]),
}

def build(seed=42):
    rnd = random.Random(seed); rows = []; i = 0
    for camp, c in C.items():
        combos = [(s, o, b, k) for s in range(3) for o in range(4) for b in range(4) for k in range(4)]
        rnd.shuffle(combos)
        for s, o, b, k in combos[:64]:
            n, d, r = rnd.choice(NAMES), rnd.choice(["Friday", "Monday", "Thursday", "the 28th"]), str(rnd.randint(10, 999))
            u = URL[c["uf"]].format(r=r)
            txt = " ".join([c["op"][o], c["bd"][b], c["cl"][k]]).format(n=n, u=u, d=d).replace("  ", " ")
            i += 1
            rows.append(dict(message_id=f"M{i:04d}", channel=c["ch"], subject_or_title=c["sub"][s], body_text=txt.strip(),
                sanitized_url_features=c["uf"], sender_domain_type=c["sd"], label=c["label"], campaign_group=camp,
                collected_date=f"2026-09-{rnd.randint(1, 28):02d}", legit_urgent=int(c["label"] == 0 and c["urg"] == 1),
                source="locally_authored_synthetic"))
    df = pd.DataFrame(rows)
    dups = df.sample(60, random_state=seed).copy()          # deliberate near-duplicates for FR-01 proof
    dups["message_id"] = [f"D{j:04d}" for j in range(len(dups))]
    dups["body_text"] = [t.replace(rnd.choice(NAMES), rnd.choice(NAMES), 1) for t in dups.body_text]
    return pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    build().to_csv("data/raw_messages.csv", index=False)
    print("wrote data/raw_messages.csv")
