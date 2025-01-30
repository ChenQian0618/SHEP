# SHEP
This is the open code of paper entitled "Shapley Estimated Explanation (Shep): A Fast Post-Hoc Attribution Method for Interpreting Intelligent Fault Diagnosis"**.

> Despite significant progress in intelligent fault diagnosis (IFD), the lack of interpretability remains a critical barrier to practical industrial applications, driving the growth of interpretability research in IFD. Post-hoc interpretability has gained popularity due to its ability to preserve network flexibility and scalability without modifying model structures. However, these methods often yield suboptimal time-domain explanations. Recently, combining domain transforms with SHAP has improved interpretability by extending explanations to more informative domains. Nonetheless, the computational expense of SHAP, exacerbated by increased dimensions from domain transforms, remains a major challenge.
>
> To address this, we propose patch-wise attribution and SHapley Estimated Explanation (SHEP). Patch-wise attribution reduces feature dimensions at the cost of explanation granularity, while SHEP simplifies subset enumeration to approximate SHAP, reducing complexity from exponential to linear. Together, these methods significantly enhance SHAP's computational efficiency, providing feasibility for real-time interpretation in monitoring tasks. Extensive experiments confirm SHEP's efficiency, interpretability, and reliability in approximating SHAP. Additionally, with open-source code, SHEP has the potential to serve as a benchmark for post-hoc interpretability in IFD.

### Notes

* **(Jan 14, 2025)**: We will upload our code after the paper is accepted.

## Contact and Related work

* chenqian2020@sjtu.edu.cn & [Homepage of Qian Chen](https://chenqian0618.github.io/Homepage/);
* CS-SHAP: ([code](https://github.com/ChenQian0618/CS-SHAP));
* TFN ([code](https://github.com/ChenQian0618/TFN) | [paper](https://www.sciencedirect.com/science/article/pii/S0888327023008609)).
