import fitz
import os

pdf_files = [
    "data/attention_paper.pdf",
    "data/CIA-RDP90-00530R000500920001-0.pdf",
    "data/enron2001.pdf",
    "data/operations_northwood.pdf",
    "data/2022_Laporan_Keuangan_1.pdf"
]

out_pdf = fitz.open()

for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        try:
            doc = fitz.open(pdf_file)
            if len(doc) > 0:
                out_pdf.insert_pdf(doc, from_page=0, to_page=0)
            doc.close()
        except Exception as e:
            print(f"Error opening {pdf_file}: {e}")

out_pdf.save("data/mixed_bag.pdf")
out_pdf.close()
print("Saved data/mixed_bag.pdf with", len(out_pdf), "pages.")
