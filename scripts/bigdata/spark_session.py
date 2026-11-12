"""Shared Spark + Apache Sedona session for the Part D notebooks (laptop or Google Colab).

Versions (requirements-bigdata.txt): PySpark 3.5.9, apache-sedona 1.9.1, with the matching jars
sedona-spark-shaded-3.5_2.12:1.9.1 and geotools-wrapper:1.9.1-33.5 (downloaded from Maven Central the first time).
On Windows, Spark also needs Hadoop's winutils.exe and hadoop.dll to write files; they are fetched once into
<project>/tools/hadoop/bin from the cdarlint/winutils repository (Hadoop 3.3.6 builds, matching Spark 3.5's
Hadoop 3.3 client) and set for this process only.
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

SEDONA_JARS = "org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.9.1,org.datasyslab:geotools-wrapper:1.9.1-33.5"
WINUTILS = "https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-3.3.6/bin/"


def windows_hadoop(project):
    """Put winutils.exe and hadoop.dll on HADOOP_HOME / PATH for this process (Windows only)."""
    home = Path(project) / "tools" / "hadoop"
    (home / "bin").mkdir(parents=True, exist_ok=True)
    for f in ("winutils.exe", "hadoop.dll"):
        if not (home / "bin" / f).exists():
            urllib.request.urlretrieve(WINUTILS + f, home / "bin" / f)
    os.environ["HADOOP_HOME"] = str(home)
    os.environ["PATH"] = str(home / "bin") + os.pathsep + os.environ["PATH"]


def get_spark(project, cores="*", memory=None):
    """Local Spark (all cores unless `cores` is given), with Sedona's spatial SQL functions registered.
    A failed task is retried up to 4 times (local[N,4]) instead of stopping the job; memory-heavy Python steps
    pass fewer cores, because each core runs its own Python worker."""
    from sedona.spark import SedonaContext

    if os.name == "nt":
        windows_hadoop(project)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    in_colab = "google.colab" in sys.modules
    # Spark's scratch folder must not contain spaces (the project path does): on Windows its block manager
    # then fails to register. Not in %TEMP% either: Windows cleaned files there during long runs (2026-11-08,
    # shuffle and pyspark temp files vanished mid-job). Colab: Colab's own fast disk.
    local_dir = Path("/content/tmp/spark") if in_colab else Path.home() / "spark-spatial-inequality"
    assert " " not in str(local_dir), f"Spark scratch folder must not contain spaces: {local_dir}"
    local_dir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(local_dir)  # PySpark's own driver-side temp files go there too
    config = (SedonaContext.builder()
              .master(f"local[{cores},4]")
              .appName("spatial-inequality")
              .config("spark.driver.memory", memory or ("8g" if in_colab else "10g"))  # Colab has ~12 GB RAM, the laptop 16 GB
              .config("spark.driver.host", "127.0.0.1")
              .config("spark.driver.bindAddress", "127.0.0.1")
              .config("spark.jars.packages", SEDONA_JARS)
              .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
              .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
              .config("spark.local.dir", str(local_dir))
              .config("spark.sql.session.timeZone", "UTC")
              .getOrCreate())
    return SedonaContext.create(config)
