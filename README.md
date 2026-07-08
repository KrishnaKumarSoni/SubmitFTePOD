<h1 align="center">SubmitFTePOD</h1>
<p align="center"><b>Drivers snap one photo of the delivery slip. AI reads off the receiver's name and number.</b></p>
<p align="center">
  <code>◐ Working</code> &nbsp;·&nbsp; Flask · OpenAI GPT-4o Vision · Vercel
</p>

> The proof of delivery already has the consignee details written on it. So instead of asking the driver to type anything, we just read the photo.

Send a driver a personal link, they get greeted by name, they upload a picture of the signed POD (proof of delivery), and GPT-4o vision pulls out the receiver's name and phone number. The driver gets one screen to confirm or fix it, then submits. No typing on a truck stop parking lot.

## Why it exists

Collecting proof of delivery from drivers is painful: they are on the road, on their phones, and rarely fill forms correctly. The one thing they always have is the physical delivery slip in hand. This tool turns that slip into structured data with a single photo, so the consignee's name and number get captured cleanly without asking a tired driver to retype what is already on the paper.

## How it works

```
Open personal link  ->  Upload POD photo  ->  GPT-4o reads the slip  ->  Confirm details  ->  Submitted
```

| Step | What happens |
|------|--------------|
| Open link | Driver opens a link with their phone number in it and is greeted by name |
| Upload | Driver snaps or picks a clear, full-page photo of the POD |
| Extract | The image is sent to GPT-4o vision, prompted to find only the receiver / consignee name and phone, and to flag anything missing |
| Confirm | Extracted name and number are shown in an editable card so the driver can correct them |
| Submit | Confirmed details are recorded and a success screen is shown |

## What it does

| Feature | What it does |
|---------|--------------|
| Photo to data | Reads the consignee name and phone straight from a delivery slip image |
| Careful extraction | Prompt is tuned to avoid mixing up driver or sender details with the receiver, and to report what is missing rather than guess |
| Personalized greeting | Recognizes the driver by the phone number in their link |
| Human in the loop | Driver reviews and edits before anything is saved |
| Mobile-first | Simple upload card built for phones, with a clear-image checklist |

## Under the hood

Flask app, OpenAI GPT-4o vision for reading the slip, deployed on Vercel.
