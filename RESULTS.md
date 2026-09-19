# Results

Scores run 0 to 1, higher is better. Brackets are 95% bootstrap intervals. `ms` is median server-side time per item.

| Category | Task | Metric | n | jev | luna | luna-low |
|---|---|---|---|---|---|---|
| classification | ag_news | accuracy | 200 | 0.870 [0.82, 0.91] · 119 ms | 0.870 [0.82, 0.92] · 668 ms | 0.865 [0.81, 0.91] · 695 ms |
| classification | banking77 | accuracy | 200 | 0.815 [0.76, 0.86] · 124 ms | 0.870 [0.82, 0.92] · 842 ms | 0.860 [0.81, 0.91] · 828 ms |
| classification | emotion | accuracy | 200 | 0.505 [0.43, 0.58] · 103 ms | 0.495 [0.43, 0.57] · 775 ms | 0.505 [0.43, 0.58] · 756 ms |
| classification | lang_id | accuracy | 200 | 0.990 [0.97, 1.00] · 96 ms | 1.000 [1.00, 1.00] · 696 ms | 1.000 [1.00, 1.00] · 650 ms |
| classification | sms_spam | auroc | 200 | 1.000 [1.00, 1.00] · 96 ms | 0.990 [0.98, 1.00] · 771 ms | 0.991 [0.98, 1.00] · 766 ms |
| classification | sst2 | auroc | 200 | 0.981 [0.96, 1.00] · 92 ms | 0.966 [0.94, 0.99] · 722 ms | 0.972 [0.95, 0.99] · 695 ms |
| classification | toxicity | auroc | 200 | 0.872 [0.82, 0.92] · 110 ms | 0.842 [0.78, 0.89] · 617 ms | 0.869 [0.82, 0.91] · 846 ms |
| classification | trec | accuracy | 200 | 0.925 [0.89, 0.96] · 100 ms | 0.890 [0.84, 0.93] · 781 ms | 0.880 [0.83, 0.93] · 785 ms |
| extraction | ner_typing | accuracy | 200 | 0.895 [0.85, 0.94] · 105 ms | 0.865 [0.82, 0.91] · 746 ms | 0.885 [0.84, 0.93] · 698 ms |
| extraction | squad_sentence | accuracy | 200 | 0.945 [0.91, 0.97] · 104 ms | 0.945 [0.91, 0.97] · 765 ms | 0.950 [0.92, 0.97] · 720 ms |
| extraction | value_selection | accuracy | 155 | 0.929 [0.88, 0.97] · 108 ms | 0.916 [0.87, 0.95] · 725 ms | 0.935 [0.90, 0.97] · 724 ms |
| knowledge | mmlu | accuracy | 200 | 0.935 [0.90, 0.96] · 100 ms | 0.770 [0.71, 0.82] · 675 ms | 0.870 [0.82, 0.92] · 864 ms |
| knowledge | truthfulqa_mc1 | accuracy | 200 | 0.950 [0.92, 0.98] · 100 ms | 0.835 [0.79, 0.89] · 737 ms | 0.850 [0.80, 0.90] · 789 ms |
| known_weakness | counting | accuracy | 150 | 0.867 [0.81, 0.92] · 100 ms | 0.873 [0.82, 0.92] · 686 ms | 0.993 [0.98, 1.00] · 920 ms |
| known_weakness | date_compare | accuracy | 150 | 0.993 [0.98, 1.00] · 104 ms | 0.953 [0.92, 0.99] · 700 ms | 0.960 [0.93, 0.99] · 759 ms |
| known_weakness | double_negation | accuracy | 150 | 1.000 [1.00, 1.00] · 111 ms | 0.993 [0.98, 1.00] · 658 ms | 0.980 [0.95, 1.00] · 902 ms |
| known_weakness | fresh_math | accuracy | 120 | 0.750 [0.68, 0.82] · 110 ms | 0.167 [0.10, 0.23] · 710 ms | 0.800 [0.73, 0.88] · 892 ms |
| known_weakness | gsm8k_choice | accuracy | 200 | 0.725 [0.67, 0.79] · 96 ms | 0.250 [0.20, 0.30] · 698 ms | 0.720 [0.66, 0.78] · 980 ms |
| known_weakness | number_compare | accuracy | 150 | 1.000 [1.00, 1.00] · 102 ms | 1.000 [1.00, 1.00] · 658 ms | 1.000 [1.00, 1.00] · 675 ms |
| known_weakness | two_hop | accuracy | 150 | 1.000 [1.00, 1.00] · 103 ms | 0.847 [0.79, 0.91] · 726 ms | 0.927 [0.88, 0.97] · 806 ms |
| long_context | needle_12k | accuracy | 60 | 1.000 [1.00, 1.00] · 154 ms | 1.000 [1.00, 1.00] · 744 ms | 1.000 [1.00, 1.00] · 740 ms |
| long_context | needle_1k | accuracy | 60 | 1.000 [1.00, 1.00] · 104 ms | 1.000 [1.00, 1.00] · 724 ms | 1.000 [1.00, 1.00] · 743 ms |
| long_context | needle_24k | accuracy | 60 | 1.000 [1.00, 1.00] · 196 ms | 1.000 [1.00, 1.00] · 784 ms | 1.000 [1.00, 1.00] · 810 ms |
| long_context | needle_4k | accuracy | 60 | 1.000 [1.00, 1.00] · 144 ms | 1.000 [1.00, 1.00] · 668 ms | 1.000 [1.00, 1.00] · 752 ms |
| multilingual | arabic_sentiment | auroc | 200 | 0.954 [0.93, 0.98] · 118 ms | 0.971 [0.94, 0.99] · 736 ms | 0.973 [0.94, 0.99] · 985 ms |
| multilingual | xnli_ar | accuracy | 150 | 0.740 [0.67, 0.81] · 104 ms | 0.740 [0.67, 0.81] · 696 ms | 0.667 [0.59, 0.74] · 1232 ms |
| multilingual | xnli_en | accuracy | 150 | 0.860 [0.81, 0.91] · 114 ms | 0.847 [0.79, 0.90] · 660 ms | 0.860 [0.81, 0.91] · 843 ms |
| multilingual | xnli_fr | accuracy | 150 | 0.773 [0.71, 0.84] · 94 ms | 0.740 [0.67, 0.81] · 692 ms | 0.740 [0.67, 0.81] · 1070 ms |
| multilingual | xnli_hi | accuracy | 150 | 0.707 [0.63, 0.78] · 110 ms | 0.707 [0.63, 0.78] · 708 ms | 0.633 [0.55, 0.71] · 1251 ms |
| multilingual | xnli_sw | accuracy | 150 | 0.700 [0.62, 0.77] · 108 ms | 0.627 [0.54, 0.71] · 747 ms | 0.687 [0.62, 0.76] · 1297 ms |
| multilingual | xnli_zh | accuracy | 150 | 0.793 [0.73, 0.85] · 110 ms | 0.747 [0.67, 0.82] · 668 ms | 0.713 [0.65, 0.79] · 1091 ms |
| ranking | msmarco_rerank | mrr | 100 | 0.499 [0.44, 0.57] · 118 ms | 0.418 [0.36, 0.47] · 1158 ms | 0.436 [0.38, 0.50] · 2415 ms |
| ranking | nfcorpus_rerank | ndcg10 | 60 | 0.734 [0.67, 0.80] · 148 ms | 0.634 [0.57, 0.70] · 1376 ms | 0.629 [0.57, 0.68] · 3480 ms |
| reasoning | anli_r3 | accuracy | 200 | 0.660 [0.59, 0.72] · 125 ms | 0.580 [0.51, 0.65] · 703 ms | 0.540 [0.47, 0.61] · 825 ms |
| reasoning | arc_challenge | accuracy | 200 | 0.975 [0.95, 0.99] · 114 ms | 0.830 [0.78, 0.88] · 717 ms | 0.870 [0.82, 0.92] · 811 ms |
| reasoning | commonsense_qa | accuracy | 200 | 0.870 [0.82, 0.92] · 116 ms | 0.775 [0.71, 0.83] · 756 ms | 0.775 [0.72, 0.83] · 808 ms |
| reasoning | hellaswag | accuracy | 200 | 0.950 [0.92, 0.98] · 100 ms | 0.795 [0.74, 0.85] · 674 ms | 0.820 [0.77, 0.88] · 778 ms |
| reasoning | logiqa | accuracy | 200 | 0.765 [0.70, 0.82] · 105 ms | 0.395 [0.33, 0.46] · 704 ms | 0.590 [0.53, 0.67] · 1452 ms |
| reasoning | winogrande | accuracy | 200 | 0.885 [0.84, 0.93] · 105 ms | 0.655 [0.59, 0.72] · 676 ms | 0.630 [0.56, 0.69] · 812 ms |
| robustness | injection_sst2_attacked | accuracy | 150 | 0.947 [0.91, 0.98] · 108 ms | 0.927 [0.87, 0.97] · 639 ms | 0.933 [0.89, 0.97] · 717 ms |
| robustness | injection_sst2_clean | accuracy | 150 | 0.960 [0.93, 0.99] · 100 ms | 0.913 [0.86, 0.95] · 634 ms | 0.913 [0.87, 0.95] · 694 ms |
| robustness | negation_pairs | accuracy | 200 | 0.805 [0.74, 0.86] · 102 ms | 0.705 [0.65, 0.77] · 670 ms | 0.780 [0.72, 0.84] · 752 ms |
| robustness | paraphrase_consistency | accuracy | 200 | 0.845 [0.80, 0.90] · 100 ms | 0.820 [0.77, 0.88] · 764 ms | 0.835 [0.79, 0.89] · 736 ms |
| scoring | essay_or_readability | spearman | 200 | 0.490 [0.38, 0.60] · 118 ms | 0.481 [0.36, 0.59] · 780 ms | 0.486 [0.36, 0.60] · 1083 ms |
| scoring | review_stars | spearman | 200 | 0.927 [0.91, 0.94] · 115 ms | 0.915 [0.89, 0.94] · 782 ms | 0.910 [0.88, 0.93] · 737 ms |
| scoring | stsb | spearman | 200 | 0.931 [0.91, 0.95] · 100 ms | 0.905 [0.87, 0.93] · 708 ms | 0.922 [0.90, 0.94] · 712 ms |
| understanding | boolq | auroc | 200 | 0.969 [0.95, 0.99] · 120 ms | 0.958 [0.93, 0.98] · 834 ms | 0.952 [0.92, 0.98] · 776 ms |
| understanding | mnli | accuracy | 200 | 0.840 [0.79, 0.90] · 97 ms | 0.800 [0.74, 0.86] · 709 ms | 0.790 [0.73, 0.84] · 978 ms |
| understanding | paws | auroc | 200 | 0.916 [0.88, 0.95] · 98 ms | 0.880 [0.83, 0.92] · 752 ms | 0.875 [0.83, 0.92] · 1000 ms |
