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
