import streamlit as st
import requests
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
import io
import os
import datetime
import subprocess
import pandas as pd
import json
import hashlib
import re
import zipfile

from pathlib import Path

# Load environment variables
load_dotenv()

# Check the environment variables
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Check if the env variables are not none before setting them
if access_key is None or secret_key is None:
    raise ValueError("MinIO credentials are empty, these need to be set to continue. Check .env file in virtual machine.")

# Set up MinIO client
minio_client = Minio(
    "10.137.0.149:9000",   # Minio Server address
    access_key=access_key,
    secret_key=secret_key,
    secure=False
)


# define buckets
bucket_name_bronze = "dw-bucket-bronze"
bucket_name_silver = "dw-bucket-silver"

def validate_filename(name):
    return bool(re.match(r'^[A-Za-z0-9 _-]+$', name))


def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(https?://)?'
        r'(([a-zA-Z0-9_-]+\.)+[a-zA-Z]{2,6})'
        r'(/[^\s]*)?$',
        re.IGNORECASE
    )
    return bool(regex.match(url.strip()))

# dataset preview & validation helpers
def _safe_size_bytes(file_obj):
    size = getattr(file_obj, "size", None)
    if size is not None:
        return size
    try:
        pos = file_obj.tell()
    except Exception:
        pos = None
    try:
        b = file_obj.read()
        size = len(b)
    finally:
        try:
            if pos is not None:
                file_obj.seek(pos)
            else:
                file_obj.seek(0)
        except Exception:
            pass
    return size or 0

def _preview_and_validate_uploaded(file_obj, name):
    """
    Show preview (CSV/JSON/XLSX) and basic validation warnings.
    Does not upload; only reads a small portion and resets pointer.
    """
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    is_tabular = ext in ("csv", "json", "xlsx")
    if not is_tabular:
        return  # Only preview/validate tabular data

    warnings = []
    preview_rows = 8

    try:
        pos = file_obj.tell()
    except Exception:
        pos = None

    try:
        if ext == "csv":
            df = pd.read_csv(file_obj, nrows=preview_rows)
        elif ext == "xlsx":
            df = pd.read_excel(file_obj, nrows=preview_rows)
        elif ext == "json":
            # Try to parse as array of records or dict; coerce to DataFrame
            raw = file_obj.read()
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    df = pd.DataFrame(data[:preview_rows])
                elif isinstance(data, dict):
                    # Flatten dict (best effort) for preview
                    df = pd.json_normalize(data)
                    df = df.head(preview_rows)
                else:
                    df = pd.DataFrame()
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not preview {name}: {e}")
        df = pd.DataFrame()
    finally:
        # Reset pointer
        try:
            if pos is not None:
                file_obj.seek(pos)
            else:
                file_obj.seek(0)
        except Exception:
            pass

    if not df.empty:
        st.caption(f"Preview of **{name}** (first {min(preview_rows, len(df))} rows):")
        st.dataframe(df, use_container_width=True)

        # Basic validation
        headers = list(df.columns)
        # Empty headers
        if any(h is None or (isinstance(h, str) and h.strip() == "") for h in headers):
            warnings.append("Empty or missing column header(s) detected.")
        # Duplicate headers
        if len(headers) != len(set(headers)):
            warnings.append("Duplicate column headers detected.")
        # Empty columns (all NaN/None/"")
        empty_cols = []
        for col in df.columns:
            series = df[col]
            # Consider NaN/None/empty string as empty
            is_empty = series.isna().all() or (series.astype(str).str.strip() == "").all()
            if is_empty:
                empty_cols.append(col)
        if empty_cols:
            warnings.append(f"Completely empty column(s): {', '.join(map(str, empty_cols))}")

        if warnings:
            for w in warnings:
                st.warning(f"{name}: {w}")
    else:
        st.info(f"No preview available for {name} or file is empty.")

    # Large file heads-up
    size_bytes = _safe_size_bytes(file_obj)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > 50:
        st.warning(f"{name}: File is large (~{size_mb:.1f} MB). Upload may take a while.")

def _parse_tags_csv(text):
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]

# Generate custom filename with suffix and prefix to enforce governance
def generate_custom_filename(project, base_name, original_filename, add_prefix_suffix):
    file_extension = original_filename.split(".")[-1]
    if add_prefix_suffix:
        date_stamp = datetime.datetime.now().strftime("%Y%m%d")
        custom_filename = f"{project}/{base_name}_{date_stamp}.{file_extension}"
    else:
        custom_filename = f"{base_name}.{file_extension}"
    return custom_filename


def upload_to_minio(file, filename, bucket_name):
    try:
        # Ensure pointer at start if previously previewed/read
        try:
            file.seek(0)
        except Exception:
            pass
        data = file.read()
        file_stream = io.BytesIO(data)
        minio_client.put_object(bucket_name, filename, file_stream, len(data))
        st.success(f"File {filename} uploaded successfully to {bucket_name}.")
    except S3Error as e:
        st.error(f"Failed to upload {filename} to {bucket_name}: {e}")


def trigger_etl(filename, preprocessing_option):
    """Trigger the ETL pipeline with the selected preprocessing option."""
    try:
        result = subprocess.run(
            ["python", "etl_pipeline.py", filename, preprocessing_option],
            check=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        st.success(f"ETL pipeline executed successfully for: {filename}")
        st.text(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to execute ETL pipeline for: {filename}")
        st.text(f"ETL Error Output: {e.stderr}")


def get_file_list(bucket):
    try:
        # Flask API to access the list of data from the VM
        api_url = f"http://10.137.0.149:5000/list-files?bucket={bucket}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to retrieve file list from {bucket}.")
            return {}
    except Exception as e:
        st.error(f"Error retrieving file list from {bucket}: {e}")
        return {}

# Function to download file using Flask API using flaskapi_dw.py
def download_file(bucket, project, filename):
    try:
        api_url = f"http://10.137.0.149:5000/download-file"
        params = {"bucket": bucket, "project": project, "filename": filename}
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Failed to download file from {bucket}. Status Code: {response.status_code}, Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error downloading file from {bucket}: {e}")
        return None

def delete_file_from_minio(bucket, object_name):
    try:
        minio_client.remove_object(bucket, object_name)
        st.success(f"Deleted {object_name} from {bucket}.")
    except S3Error as e:
        st.error(f"Failed to delete {object_name} from {bucket}: {e}")

def main():
    st.title("File Upload and Download for Redback Data Warehouse")

    # Initialize session state
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []

    
    # Create tabs for File Upload, Bronze, and Silver
    tabs = st.tabs(["File Upload & ETL", "View Original Files", "View Pre-processed Files"])

    #  Tab 1: File Upload & ETL
    with tabs[0]:
        st.header("File Upload Section")

        project = st.selectbox("Select Project", options=["project1", "project2", "project3", "project4", "project5","other"])
        preprocessing = st.selectbox("Preprocessing (optional)", options=["No Pre-processing", "Data Clean Up", "Preprocessing for Machine Learning"])
        add_prefix = st.checkbox("Add project as prefix and date as suffix to filename (to overwrite existing files)", value=True)

        provenance_source = st.text_input("Provenance Source (e.g., Wikipedia, Internal DB, Kaggle)")
        source_url = st.text_input("Source URL (optional)")

        # bulk tags for all files
        bulk_tags_text = st.text_input("Tags for all files (comma-separated, optional)")
        bulk_tags = _parse_tags_csv(bulk_tags_text)

        st.markdown("### File Upload (Multiple Files or ZIP Supported)")
        uploaded_items = st.file_uploader(
            "Upload files (csv, txt, json, xlsx, images, videos) or a zip containing datasets",
            type=["csv", "txt", "json", "xlsx", "zip", "jpg", "jpeg", "png", "gif", "mp4", "avi", "mov"],
            accept_multiple_files=True
        )

        file_metadata = []
        zip_files = []

        if uploaded_items:
            # Show previews/validation for tabular files and collect per-file base/override tags
            for file in uploaded_items:
                if file.name.lower().endswith(".zip"):
                    zip_files.append(file)
                    # ZIP preview feature
                    st.markdown(f"**Contents of ZIP: {file.name}**")
                    try:
                        with zipfile.ZipFile(file, "r") as zip_ref:
                            file_list = zip_ref.namelist()
                            st.write(file_list)
                            # Preview tabular files inside ZIP
                            for fname in file_list:
                                ext = fname.lower().rsplit(".", 1)[-1]
                                if ext in ("csv", "json", "xlsx"):
                                    st.caption(f"Preview of `{fname}` in ZIP:")
                                    try:
                                        with zip_ref.open(fname) as f:
                                            if ext == "csv":
                                                df = pd.read_csv(f, nrows=8)
                                            elif ext == "xlsx":
                                                df = pd.read_excel(f, nrows=8)
                                            elif ext == "json":
                                                raw = f.read()
                                                try:
                                                    data = json.loads(raw)
                                                    if isinstance(data, list):
                                                        df = pd.DataFrame(data[:8])
                                                    elif isinstance(data, dict):
                                                        df = pd.json_normalize(data)
                                                        df = df.head(8)
                                                    else:
                                                        df = pd.DataFrame()
                                                except Exception:
                                                    df = pd.DataFrame()
                                            else:
                                                df = pd.DataFrame()
                                            if not df.empty:
                                                st.dataframe(df, use_container_width=True)
                                            else:
                                                st.info(f"No preview available for {fname} or file is empty.")
                                    except Exception as e:
                                        st.warning(f"Could not preview {fname} in ZIP: {e}")
                    except Exception as e:
                        st.warning(f"Could not open ZIP file {file.name}: {e}")

                else:
                    default_base = file.name.rsplit('.', 1)[0]
                    base = st.text_input(f"Base name for {file.name}", value=default_base, key=f"base_{file.name}")
                    # per-file tags override
                    per_file_tags_text = st.text_input(f"Tags for {file.name} (comma-separated, optional)", value="", key=f"tags_{file.name}")
                    per_file_tags = _parse_tags_csv(per_file_tags_text)
                    file_metadata.append({"file": file, "base": base, "tags": per_file_tags if per_file_tags else bulk_tags})
                    # Preview + validate datasets (CSV/JSON/XLSX)
                    _preview_and_validate_uploaded(file, file.name)

        apply_bulk = st.checkbox("Apply provenance & preprocessing to all uploaded files", value=True)

        st.markdown("---")
        if st.button("Upload Files"):

            if not uploaded_files:
                st.warning("Please select at least one file.")
            elif not valid_basenames:
                st.warning("Please fix invalid base names.")
            if not uploaded_items:
                st.warning("Please select at least one file or zip.")
            elif provenance_source.strip() == "":
                st.warning("Please enter a provenance source before uploading.")
            elif source_url and not is_valid_url(source_url):
                st.warning("The Source URL format appears to be invalid. Please enter a valid URL.")
            else:
                st.session_state.uploaded_filenames = []
                for idx, file in enumerate(uploaded_files):
                    custom_name = generate_custom_filename(project, base_names[idx], file.name, add_prefix)

                # progress bar for real-time feedback
                total_to_upload = len(file_metadata)
                for zf in zip_files:
                    try:
                        with zipfile.ZipFile(zf, "r") as zip_ref:
                            # count files eligible in each zip
                            for fname in zip_ref.namelist():
                                if fname.lower().endswith((".csv", ".txt", ".json", ".xlsx", ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".avi", ".mov")):
                                    total_to_upload += 1
                    except Exception:
                        pass

                progress = st.progress(0)
                completed = 0

                # Handle regular files (including images/videos)
                for meta in file_metadata:
                    base = meta["base"]
                    file = meta["file"]
                    tags_for_file = meta["tags"]

                    if not validate_filename(base):
                        st.warning(f"Base name for {file.name} must be alphanumeric.")
                        completed += 1
                        progress.progress(min(int((completed/total_to_upload)*100), 100))
                        continue

                    custom_name = generate_custom_filename(project, base, file.name, add_prefix)

                    # Upload file to MinIO
                    upload_to_minio(file, custom_name, bucket_name_bronze)
                    st.session_state.uploaded_filenames.append(custom_name)

                    # Provenance entry
                    new_entry = {
                        "upload_time": datetime.datetime.now().isoformat(),
                        "provenance_source": provenance_source if apply_bulk else "",
                        "source_url": source_url if apply_bulk else "",
                        "preprocessing": preprocessing if apply_bulk else "",
                        "uploaded_by": os.getenv("USERNAME") or os.getenv("USER") or "unknown",
                        "file_type": file.type,
                        "tags": tags_for_file 
                    }
                    entry_str = json.dumps(new_entry, sort_keys=True)
                    new_entry["signature"] = hashlib.sha256(entry_str.encode("utf-8")).hexdigest()

                    provenance = {
                        "filename": custom_name,
                        "original_filename": file.name,
                        "project": project,
                        "bucket": bucket_name_bronze,
                        "history": [new_entry]
                    }

                    upload_provenance_to_minio(provenance, custom_name, bucket_name_bronze)

                    # preview in app
                    if file.type.startswith("image/"):
                        st.image(file)
                    elif file.type.startswith("video/"):
                        st.video(file)

                    completed += 1
                    progress.progress(min(int((completed/total_to_upload)*100), 100))

                # Handle zip uploads (datasets + images/videos)
                for zf in zip_files:
                    with zipfile.ZipFile(zf, "r") as zip_ref:
                        for fname in zip_ref.namelist():
                            if fname.lower().endswith((".csv", ".txt", ".json", ".xlsx", ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".avi", ".mov")):
                                data = zip_ref.read(fname)
                                base = fname.rsplit('.', 1)[0]
                                if not validate_filename(base):
                                    st.warning(f"Base name for {fname} must be alphanumeric. Skipping.")
                                    completed += 1
                                    progress.progress(min(int((completed/total_to_upload)*100), 100))
                                    continue

                                custom_name = generate_custom_filename(project, base, fname, add_prefix)
                                upload_to_minio(io.BytesIO(data), custom_name, bucket_name_bronze)
                                st.session_state.uploaded_filenames.append(custom_name)

                                # Guess file type from extension
                                ext = fname.rsplit('.', 1)[-1].lower()
                                file_type = {
                                    "csv": "text/csv",
                                    "txt": "text/plain",
                                    "json": "application/json",
                                    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    "jpg": "image/jpeg",
                                    "jpeg": "image/jpeg",
                                    "png": "image/png",
                                    "gif": "image/gif",
                                    "mp4": "video/mp4",
                                    "avi": "video/x-msvideo",
                                    "mov": "video/quicktime"
                                }.get(ext, "application/octet-stream")

                                new_entry = {
                                    "upload_time": datetime.datetime.now().isoformat(),
                                    "provenance_source": provenance_source,
                                    "source_url": source_url,
                                    "preprocessing": preprocessing,
                                    "uploaded_by": os.getenv("USERNAME") or os.getenv("USER") or "unknown",
                                    "file_type": file_type,
                                    "tags": bulk_tags  # bulk tags apply to zip contents
                                }
                                entry_str = json.dumps(new_entry, sort_keys=True)
                                new_entry["signature"] = hashlib.sha256(entry_str.encode("utf-8")).hexdigest()

                                provenance = {
                                    "filename": custom_name,
                                    "original_filename": fname,
                                    "project": project,
                                    "bucket": bucket_name_bronze,
                                    "history": [new_entry]
                                }
                                upload_provenance_to_minio(provenance, custom_name, bucket_name_bronze)

                                completed += 1
                                progress.progress(min(int((completed/total_to_upload)*100), 100))

        # Option to trigger ETL after all uploads
        if st.session_state.uploaded_filenames:
            if st.button("Trigger ETL for All Uploaded Files"):
                for filename in st.session_state.uploaded_filenames:
                    trigger_etl(filename, preprocessing)
        
     # Tab 2: View Bronze Files
    with tabs[1]:
        st.header("Uploaded Files Overview - Bronze (dw-bucket-bronze)")
        # Get the list of files from the "dw-bucket-bronze" bucket
        files_by_project = get_file_list("dw-bucket-bronze")

        # quick filename filter
        name_filter = st.text_input("Filter by filename (contains)", value="")

        if files_by_project:
            available_projects = list(files_by_project.keys())  # Get project names (folders)
            selected_project = st.selectbox("Select Project Folder", available_projects)

            if selected_project in files_by_project:
                items = files_by_project[selected_project]
                if name_filter.strip():
                    items = [f for f in items if name_filter.strip().lower() in f.lower()]

                file_list = [{"Project": selected_project, "File": file} for file in items]

                if file_list:
                    df = pd.DataFrame(file_list)
                    st.dataframe(df)  # Display the table with the filtered list of files

                    if items:
                        # Bulk select
                        selected_files = st.multiselect("Select File(s) to Download/Delete", [row["File"] for row in file_list], key="bronze_bulk_select")

                    if st.button("Download Selected File from Bronze"):
                        file_content = download_file("dw-bucket-bronze", selected_project, selected_file)
                        if file_content:
                            st.download_button(label=f"Download {selected_file}", data=file_content, file_name=selected_file.split("/")[-1])
    
     
                        # Download single file (keep for convenience)
                        if len(selected_files) == 1 and st.button("Download Selected File from Bronze"):
                            file_content = download_file("dw-bucket-bronze", selected_project, selected_files[0])
                            if file_content:
                                st.download_button(label=f"Download {selected_files[0]}", data=file_content, file_name=selected_files[0].split("/")[-1])

                        # Confirmation and bulk delete
                        if selected_files:
                            confirm = st.checkbox("Confirm deletion of selected file(s) from Bronze", key="bronze_confirm")
                            if confirm and st.button("Delete Selected File(s) from Bronze"):
                                for f in selected_files:
                                    delete_file_from_minio("dw-bucket-bronze", f"{selected_project}/{f}")
                                st.success(f"Deleted {len(selected_files)} file(s) from Bronze.")
                                st.experimental_rerun()
    # Tab 3: View Silver Files
    with tabs[2]:
        st.header("Uploaded Files Overview - Silver (dw-bucket-silver)")
        # Get the list of files from the "dw-bucket-silver" bucket
        files_by_project = get_file_list("dw-bucket-silver")

        # quick filename filter 
        name_filter_silver = st.text_input("Filter by filename (contains) - Silver", value="", key="silver_filter")

        if files_by_project:
            available_projects = list(files_by_project.keys())  # Get project names (folders)
            selected_project = st.selectbox("Select Project Folder", available_projects)

            if selected_project in files_by_project:
                items = files_by_project[selected_project]
                if name_filter_silver.strip():
                    items = [f for f in items if name_filter_silver.strip().lower() in f.lower()]

                file_list = [{"Project": selected_project, "File": file} for file in items]

                if file_list:
                    df = pd.DataFrame(file_list)
                    st.dataframe(df)  # Display the table with the filtered list of files

                    selected_file = st.selectbox("Select File to Download", df["File"].tolist())
                    if items:
                        selected_files = st.multiselect("Select File(s) to Download/Delete", [row["File"] for row in file_list], key="silver_bulk_select")

                    if st.button("Download Selected File from Silver"):
                        file_content = download_file("dw-bucket-silver", selected_project, selected_file)
                        if file_content:
                            st.download_button(label=f"Download {selected_file}", data=file_content, file_name=selected_file.split("/")[-1])
            
                        if len(selected_files) == 1 and st.button("Download Selected File from Silver"):
                            file_content = download_file("dw-bucket-silver", selected_project, selected_files[0])
                            if file_content:
                                st.download_button(label=f"Download {selected_files[0]}", data=file_content, file_name=selected_files[0].split("/")[-1])

                        if selected_files:
                            confirm = st.checkbox("Confirm deletion of selected file(s) from Silver", key="silver_confirm")
                            if confirm and st.button("Delete Selected File(s) from Silver"):
                                for f in selected_files:
                                    delete_file_from_minio("dw-bucket-silver", f"{selected_project}/{f}")
                                st.success(f"Deleted {len(selected_files)} file(s) from Silver.")
                                st.experimental_rerun()
    # Tab 4: View Provenance Logs
    with tabs[3]:
        st.header("Provenance Logs (MinIO)")
        try:
            objects = minio_client.list_objects(bucket_name_bronze, recursive=True)
            provenance_files = [obj.object_name for obj in objects if obj.object_name.endswith(".provenance.json")]
        except S3Error as e:
            provenance_files = []
            st.error(f"Failed to list provenance files: {e}")

        # Tag search across provenance
        tag_query = st.text_input("Search by tag (exact match)", value="")
        results = []

        if provenance_files:
            if tag_query.strip():
                # Scan provenance for matching tags
                for prov_key in provenance_files:
                    try:
                        response = minio_client.get_object(bucket_name_bronze, prov_key)
                        data = json.load(response)
                        hist = data.get("history", [])
                        tags = []
                        # collect tags from last entry (current convention: single entry list)
                        if hist:
                            tags = hist[-1].get("tags", []) or []
                        if tag_query.strip() in tags:
                            results.append({
                                "Provenance": prov_key,
                                "Filename": data.get("filename", ""),
                                "Original": data.get("original_filename", ""),
                                "Tags": ", ".join(tags)
                            })
                    except Exception:
                        # ignore malformed provenance files
                        continue

                if results:
                    st.subheader("Files matching tag:")
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.info("No files matched that tag.")

            selected_log = st.selectbox("Or select a provenance log to view", provenance_files)
            if selected_log:
                try:
                    response = minio_client.get_object(bucket_name_bronze, selected_log)
                    data = json.load(response)
                    st.json(data)
                    if st.button("Delete This Provenance Log"):
                        delete_file_from_minio(bucket_name_bronze, selected_log)
                except Exception as e:
                    st.error(f"Failed to load provenance log: {e}")

        if provenance_files:
            # Bulk select
            selected_logs = st.multiselect("Select Provenance Log(s) to View/Delete", provenance_files, key="prov_bulk_select")

            if selected_logs:
                # Show first selected log for preview
                try:
                    response = minio_client.get_object(bucket_name_bronze, selected_logs[0])
                    data = json.load(response)
                    st.json(data)
                except Exception as e:
                    st.error(f"Failed to load provenance log: {e}")

                confirm = st.checkbox("Confirm deletion of selected provenance log(s)", key="prov_confirm")
                if confirm and st.button("Delete Selected Provenance Log(s)"):
                    for log in selected_logs:
                        delete_file_from_minio(bucket_name_bronze, log)
                    st.success(f"Deleted {len(selected_logs)} provenance log(s).")
                    st.experimental_rerun()
        else:
            st.info("No provenance logs found in MinIO.")

if __name__ == "__main__":
    main()
