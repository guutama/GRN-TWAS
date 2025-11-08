```
### Gene Regulatory Network-Driven Transcriptome-Wide Association Studies (GRN-TWAS)

This repository contains the implementation of a novel framework that integrates **tissue-specific gene regulatory networks (GRNs)** into **transcriptome-wide association studies (TWAS)** for studying gene-complex disease associations.

### Key Features
- Utilizes **Findr** for GRN reconstruction from genotype and transcriptome data.
- Predicts gene expression by incorporating both **cis** and **trans** regulatory components.
- Evaluates gene-disease associations using **GWAS summary statistics**.

### Data Sources
This project uses three main data sources:


---
1. **Reference dataset of genotype and gene expression**  
   Individual-level genotype data and RNA sequencing-based gene expression profiles for the same individuals.


      ```plaintext
      gene_id     sample1   sample2   sample3   sample4   sample5
      gene1  0.415     0.167     0.000     0.034     0.884
      gene2  0.614     0.000     0.506     0.852     0.000
      gene5  0.827     0.000     0.329     0.889     0.179
      gene3  1.229     0.248     0.000     0.000     0.295
      gene4  0.832     0.000     0.644     0.495     0.437
      ```

      ```plaintext
      snp_id   chromosome  position   ref  alt  sample1  sample2  sample3  sample4  sample5 
      rs1         6           4162        C    G    0        0        1        0        1 
      rs2         16          9857        G    A    0        0        1        0        0 
      rs5         3           1603        A    G    0        0        0        0        1 
      rs3         4           1470        T    C    2        0        1        0        1 
      snp4        5           3608        C    T    2        0        0        1        2 
      ```

2. **eQTL mapping summary statistics**
   Summary statistics from expression quantitative trait locus (eQTL) mapping that link SNPs to genes based on their regulatory effects. The data includes:

   - **SNP_ID**: Identifier for the genetic variant.
   - **GENE_ID**: Associated gene identifier.
   - **Beta**: Effect size of the SNP on gene expression.

      ```plaintext
      snp_id   gene_id  beta    adj.p-value
      snp1     gene1    0.35      0.005497
      snp2     gene2   -0.28     0.00479
      snp5     gene5    0.48      0.000500
      snp3     gene3   -0.26     0.045477
      snp4     gene4    0.37     0.000500
      ...
      ```

3. **Genome-wide summary statistics**
   GWAS meta-analysis results, which include data on millions of genetic variants with information such as:

   - **snpid**: Identifier for genetic variant.
   - **logOR**: Log odds ratio representing effect sizes. If other statistics are used (e.g., beta values), adjust the association code accordingly.
   - **se_gc**: The standard error of the estimated effect size.

      ```plaintext
      snp_id     beta      se_beta
      rs5         0.050       0.27
      rs1         0.005       0.01
      rs3         0.003       0.01
      rs2         0.001       0.01
      rs6         0.003       0.01
      ...           ...        ...
      ```










<!-- ### GRN-TWAS Pipeline

This repository includes a Julia pipeline that automates the three main stages of our framework:

1. **Network Reconstruction**: 
   - Reconstructs tissue-specific gene regulatory networks (GRNs) using reference genotype and gene expression data.
   - Leverages tools like `Findr` for causal inference to build GRNs.

2. **Model Training**: 
   - Trains a machine learning model (e.g., Ridge regression) to predict gene expression by incorporating cis- and trans-eQTL regulatory effects derived from GRNs.

3. **Association Analysis**: 
   - Integrates GWAS summary statistics with predicted gene expression to evaluate gene-disease associations.

### Running the Pipeline

To execute the full pipeline, ensure the following inputs are prepared: -->
<!-- - **Reference Dataset**: A file containing genotype and gene expression data (`reference_dataset.tsv`).
- **GWAS Summary Statistics**: A file with genome-wide summary statistics (`gwas_summary_statistics.tsv`). -->

<!-- Run the pipeline script:

```bash
python ../src/grn_gwas_main.py 
``` -->




<!-- ### Input Format
The input file must be a gzipped CSV file (`.csv.gz`) containing the following columns:
 -->

<!-- 




1. **id**  
   - Unique identifier for each gene or SNP.

2. **Expression Columns**  
   - Columns representing expression data for each sample.
   - Column names should be unique and represent sample identifiers (e.g., `sample1`, `sample2`, etc.).

3. **Genotype Columns**  
   - Columns representing genotype data for each sample.
   - Column names should match the expression sample identifiers.

### Example Input File
```csv
id,sample1,sample2,sample3
gene1,5.6,3.2,4.1
gene2,4.1,2.1,5.2
```

```csv
id,sample1,sample2,sample3
snp1,0,1,0
snp2,1,0,1
``` -->
```
