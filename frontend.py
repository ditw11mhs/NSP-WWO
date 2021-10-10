import os
import numpy as np
from backend import NSP_Class, WWO
import streamlit as st
import altair as alt
import pandas as pd


class Main:
    def __init__(self):
        pass

    def main(self):
        st.set_page_config(
            page_title="Optimizing Nurse Schedulin Problem with Water Wave Optimization",
            page_icon=":calendar:",
            layout="wide",
        )
        # Input
        self.input()
        # Output
        self.output()

    def input(self):
        st.session_state["optimize"] = False
        with st.sidebar.form("Input"):
            st.title("Input Parameter")
            st.number_input("Population", key="x_population", value=10)
            st.number_input("Iteration", key="iter", value=10)
            st.number_input("Hard Constraint Multiplier", key="multiplier", value=0)
            st.number_input(
                "H Max (Max Wave Height)",
                key="hmax",
                value=6.0,
                step=0.000001,
                max_value=100000.0,
                min_value=0.0,
                format="%f",
            )
            st.number_input(
                "Lambda (Wavelength)",
                key="lambd",
                value=0.5,
                step=0.000001,
                max_value=100000.0,
                min_value=0.0,
                format="%f",
            )
            st.number_input(
                "Alpha",
                key="alpha",
                value=1.001,
                step=0.000001,
                max_value=100000.0,
                min_value=0.0,
                format="%f",
            )
            st.number_input(
                "Beta Max",
                key="beta_max",
                value=0.01,
                step=0.000001,
                max_value=100000.0,
                min_value=0.0,
                format="%f",
            )
            st.number_input(
                "Beta Min",
                key="beta_min",
                value=0.001,
                step=0.000001,
                max_value=100000.0,
                min_value=0.0,
                format="%f",
            )
            st.number_input(
                "Epsilon",
                key="epsilon",
                value=1e-31,
                format="%f",
                step=1e-31,
                min_value=0.0,
                max_value=1.0,
            )
            st.number_input("K Max", key="k_max", value=12)
            st.number_input("Upper Bound", key="upper_bound", value=3)
            st.number_input("Lower Bound", key="lower_bound", value=0)
            if st.form_submit_button("Optimize"):
                st.session_state["optimize"] = True

    def output(self):
        st.title("Optimizing Nurse Scheduling Problem with Water Wave Optimization")
        units_name = ["IGD", "Rawat Inap", "Anastesi", "ICU", "OK"]
        units_nurse_num = np.array([30, 122, 23, 28, 33])
        units_morning_shift = np.array([8, 34, 6, 7, 8])
        units_afternoon_shift = np.array([8, 34, 6, 7, 8])
        units_night_shift = np.array([6, 24, 5, 6, 6])

        units_minimum_shift = np.vstack(
            (units_morning_shift, units_afternoon_shift, units_night_shift)
        ).T
        units_minimum_shift = np.hstack((units_minimum_shift, np.zeros((5, 1))))

        cols = st.columns(len(units_name))
        for index in range(len(units_name)):
            with cols[index]:
                st.header(units_name[index])
                st.write(
                    f"""Total Nurse = {str(units_nurse_num[index])}  \nMorning Shift = {str(units_morning_shift[index])}  \nNoon Shift = {str(units_afternoon_shift[index])}  \nNight Shift = {str(units_night_shift[index])}"""
                )

        if st.session_state["optimize"]:
            NSP_Class_Dict = {}
            for index, unit in enumerate(units_name):
                NSP_Class_Dict[unit] = NSP_Class(
                    day=30,
                    units_name=unit,
                    unit_total_nurse=units_nurse_num[index],
                    unit_minimum_shift=units_minimum_shift[index, :],
                    hard_constraint_multiplier=st.session_state["multiplier"],
                )
            cols2 = st.columns(3)
            with cols2[0]:
                st.subheader("IGD")
                st.dataframe(
                    pd.DataFrame(
                        NSP_Class_Dict["IGD"].nurse_array_col,
                        index=[
                            "Perawat " + str(i)
                            for i in np.arange(
                                1, NSP_Class_Dict["IGD"].nurse_array_col.shape[0] + 1
                            )
                        ],
                        columns=[
                            "Hari " + str(i)
                            for i in np.arange(
                                1, NSP_Class_Dict["IGD"].nurse_array_col.shape[1] + 1
                            )
                        ],
                    )
                )
                st.subheader("Rawat Inap")
                st.dataframe(
                    pd.DataFrame(
                        NSP_Class_Dict["Rawat Inap"].nurse_array_col,
                        index=[
                            "Perawat " + str(i)
                            for i in np.arange(
                                1,
                                NSP_Class_Dict["Rawat Inap"].nurse_array_col.shape[0]
                                + 1,
                            )
                        ],
                        columns=[
                            "Hari " + str(i)
                            for i in np.arange(
                                1,
                                NSP_Class_Dict["Rawat Inap"].nurse_array_col.shape[1]
                                + 1,
                            )
                        ],
                    )
                )
                st.subheader("Anastesi")
                st.dataframe(
                    pd.DataFrame(
                        NSP_Class_Dict["Anastesi"].nurse_array_col,
                        index=[
                            "Perawat " + str(i)
                            for i in np.arange(
                                1,
                                NSP_Class_Dict["Anastesi"].nurse_array_col.shape[0] + 1,
                            )
                        ],
                        columns=[
                            "Hari " + str(i)
                            for i in np.arange(
                                1,
                                NSP_Class_Dict["Anastesi"].nurse_array_col.shape[1] + 1,
                            )
                        ],
                    )
                )
                st.subheader("ICU")
                st.dataframe(
                    pd.DataFrame(
                        NSP_Class_Dict["ICU"].nurse_array_col,
                        index=[
                            "Perawat " + str(i)
                            for i in np.arange(
                                1, NSP_Class_Dict["ICU"].nurse_array_col.shape[0] + 1
                            )
                        ],
                        columns=[
                            "Hari " + str(i)
                            for i in np.arange(
                                1, NSP_Class_Dict["ICU"].nurse_array_col.shape[1] + 1
                            )
                        ],
                    )
                )
                st.subheader("OK")
                st.dataframe(
                    pd.DataFrame(
                        NSP_Class_Dict["OK"].nurse_array_col,
                        index=[
                            "Perawat " + str(i)
                            for i in np.arange(
                                1, NSP_Class_Dict["OK"].nurse_array_col.shape[0] + 1
                            )
                        ],
                        columns=[
                            "Hari " + str(i)
                            for i in np.arange(
                                1, NSP_Class_Dict["OK"].nurse_array_col.shape[1] + 1
                            )
                        ],
                    )
                )

            with cols2[1]:
                igd_plot_text = st.empty()
                igd_plot = st.empty()
                r_inap_plot_text = st.empty()
                r_inap_plot = st.empty()
                anastesi_plot_text = st.empty()
                anastesi_plot = st.empty()
                icu_plot_text = st.empty()
                icu_plot = st.empty()
                ok_plot_text = st.empty()
                ok_plot = st.empty()

            with cols2[2]:
                igd_cost_text = st.empty()
                igd_2 = st.empty()
                r_inap_cost_text = st.empty()
                r_inap_2 = st.empty()
                anastesi_cost_text = st.empty()
                anastesi_2 = st.empty()
                icu_cost_text = st.empty()
                icu_2 = st.empty()
                ok_cost_text = st.empty()
                ok_2 = st.empty()

            WWO_Class_Dict = {}
            for index, unit_name in enumerate(units_name):
                WWO_Class_Dict[unit_name] = WWO(
                    NSP=NSP_Class_Dict[unit_name],
                    iteration=st.session_state["iter"],
                    hmax=st.session_state["hmax"],
                    lambd=st.session_state["lambd"],
                    alpha=st.session_state["alpha"],
                    epsilon=st.session_state["epsilon"],
                    beta_max=st.session_state["beta_max"],
                    beta_min=st.session_state["beta_min"],
                    k_max=st.session_state["k_max"],
                    upper_bound=st.session_state["upper_bound"],
                    lower_bound=st.session_state["lower_bound"],
                    x_population=st.session_state["x_population"],
                )
            with cols2[1]:
                with st.spinner("Processing IGD Schedule..."):
                    igd_pos, _ = WWO_Class_Dict["IGD"].optimize()
                    igd_plot_text.subheader("IGD WWO Plot")
                    data_igd = pd.DataFrame(
                        {
                            "Cost Jadwal 2 Unit IGD": WWO_Class_Dict[
                                "IGD"
                            ].best_fit_iteration,
                            "Iteration": np.arange(st.session_state["iter"] + 1),
                        }
                    )
                    chart_igd = (
                        alt.Chart(data_igd)
                        .mark_line()
                        .encode(x="Iteration", y="Cost Jadwal 2 Unit IGD")
                    )
                    igd_plot.altair_chart(chart_igd, use_container_width=True)

                    igd_cost_text.subheader(
                        f"""Cost {max(WWO_Class_Dict["IGD"].best_fit_iteration)} -> {min(WWO_Class_Dict["IGD"].best_fit_iteration)}"""
                    )
                    igd_2.dataframe(
                        pd.DataFrame(
                            igd_pos,
                            index=[
                                "Perawat " + str(i)
                                for i in np.arange(1, igd_pos.shape[0] + 1)
                            ],
                            columns=[
                                "Hari " + str(i)
                                for i in np.arange(1, igd_pos.shape[1] + 1)
                            ],
                        )
                    )
                # st.success()

                with st.spinner("Processing Rawat Inap Schedule..."):
                    r_inap_pos, _ = WWO_Class_Dict["Rawat Inap"].optimize()
                    r_inap_plot_text.subheader("Rawat Inap WWO Plot")
                    data_r_inap = pd.DataFrame(
                        {
                            "Cost Jadwal 2 Unit Rawat Inap": WWO_Class_Dict[
                                "Rawat Inap"
                            ].best_fit_iteration,
                            "Iteration": np.arange(st.session_state["iter"] + 1),
                        }
                    )
                    chart_r_inap = (
                        alt.Chart(data_r_inap)
                        .mark_line()
                        .encode(x="Iteration", y="Cost Jadwal 2 Unit Rawat Inap")
                    )
                    r_inap_plot.altair_chart(chart_r_inap, use_container_width=True)
                    r_inap_cost_text.subheader(
                        f"""Cost {max(WWO_Class_Dict["Rawat Inap"].best_fit_iteration)} -> {min(WWO_Class_Dict["Rawat Inap"].best_fit_iteration)}"""
                    )
                    r_inap_2.dataframe(
                        pd.DataFrame(
                            r_inap_pos,
                            index=[
                                "Perawat " + str(i)
                                for i in np.arange(1, r_inap_pos.shape[0] + 1)
                            ],
                            columns=[
                                "Hari " + str(i)
                                for i in np.arange(1, r_inap_pos.shape[1] + 1)
                            ],
                        )
                    )
                with st.spinner("Processing Anastesi Schedule..."):
                    anastesi_pos, _ = WWO_Class_Dict["Anastesi"].optimize()
                    anastesi_plot_text.subheader("Anastesi WWO Plot")
                    data_anastesi = pd.DataFrame(
                        {
                            "Cost Jadwal 2 Unit Anastesi": WWO_Class_Dict[
                                "Anastesi"
                            ].best_fit_iteration,
                            "Iteration": np.arange(st.session_state["iter"] + 1),
                        }
                    )
                    chart_anastesi = (
                        alt.Chart(data_anastesi)
                        .mark_line()
                        .encode(x="Iteration", y="Cost Jadwal 2 Unit Anastesi")
                    )
                    anastesi_plot.altair_chart(chart_anastesi, use_container_width=True)
                    anastesi_cost_text.subheader(
                        f"""Cost {max(WWO_Class_Dict["Anastesi"].best_fit_iteration)} -> {min(WWO_Class_Dict["Anastesi"].best_fit_iteration)}"""
                    )
                    anastesi_2.dataframe(
                        pd.DataFrame(
                            anastesi_pos,
                            index=[
                                "Perawat " + str(i)
                                for i in np.arange(1, anastesi_pos.shape[0] + 1)
                            ],
                            columns=[
                                "Hari " + str(i)
                                for i in np.arange(1, anastesi_pos.shape[1] + 1)
                            ],
                        )
                    )
                with st.spinner("Processing ICU Schedule..."):
                    icu_pos, _ = WWO_Class_Dict["ICU"].optimize()
                    icu_plot_text.subheader("ICU WWO Plot")
                    data_icu = pd.DataFrame(
                        {
                            "Cost Jadwal 2 Unit ICU": WWO_Class_Dict[
                                "ICU"
                            ].best_fit_iteration,
                            "Iteration": np.arange(st.session_state["iter"] + 1),
                        }
                    )
                    chart_icu = (
                        alt.Chart(data_icu)
                        .mark_line()
                        .encode(x="Iteration", y="Cost Jadwal 2 Unit ICU")
                    )
                    icu_plot.altair_chart(chart_icu, use_container_width=True)
                    icu_cost_text.subheader(
                        f"""Cost {max(WWO_Class_Dict["ICU"].best_fit_iteration)} -> {min(WWO_Class_Dict["ICU"].best_fit_iteration)}"""
                    )
                    icu_2.dataframe(
                        pd.DataFrame(
                            icu_pos,
                            index=[
                                "Perawat " + str(i)
                                for i in np.arange(1, icu_pos.shape[0] + 1)
                            ],
                            columns=[
                                "Hari " + str(i)
                                for i in np.arange(1, icu_pos.shape[1] + 1)
                            ],
                        )
                    )
                with st.spinner("Processing OK Schedule..."):
                    ok_pos, _ = WWO_Class_Dict["OK"].optimize()
                    ok_plot_text.subheader("OK WWO Plot")
                    data_ok = pd.DataFrame(
                        {
                            "Cost Jadwal 2 Unit OK": WWO_Class_Dict[
                                "OK"
                            ].best_fit_iteration,
                            "Iteration": np.arange(st.session_state["iter"] + 1),
                        }
                    )
                    chart_ok = (
                        alt.Chart(data_ok)
                        .mark_line()
                        .encode(x="Iteration", y="Cost Jadwal 2 Unit OK")
                    )
                    ok_plot.altair_chart(chart_ok, use_container_width=True)
                    ok_cost_text.subheader(
                        f"""Cost {max(WWO_Class_Dict["OK"].best_fit_iteration)} -> {min(WWO_Class_Dict["OK"].best_fit_iteration)}"""
                    )
                    ok_2.dataframe(
                        pd.DataFrame(
                            ok_pos,
                            index=[
                                "Perawat " + str(i)
                                for i in np.arange(1, ok_pos.shape[0] + 1)
                            ],
                            columns=[
                                "Hari " + str(i)
                                for i in np.arange(1, ok_pos.shape[1] + 1)
                            ],
                        )
                    )


if __name__ == "__main__":
    Main().main()
