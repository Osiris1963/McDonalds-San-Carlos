with tabs[0]:
            if not st.session_state.forecast_df.empty:
                today=pd.to_datetime('today').normalize();future_forecast_df=st.session_state.forecast_df[st.session_state.forecast_df['ds']>=today].copy()
                if future_forecast_df.empty:st.warning("Forecast contains no future dates.")
                else:
                    disp_cols={'ds':'Date','forecast_customers':'Predicted Customers','forecast_atv':'Predicted Avg Sale (₱)','forecast_sales':'Predicted Sales (₱)','weather':'Predicted Weather'}
                    existing_disp_cols={k:v for k,v in disp_cols.items()if k in future_forecast_df.columns};display_df=future_forecast_df.rename(columns=existing_disp_cols);final_cols_order=[v for k,v in disp_cols.items()if k in existing_disp_cols]
                    st.markdown("#### 15-Day Forecast");st.dataframe(display_df[final_cols_order].set_index('Date').style.format({'Predicted Customers':'{:,.0f}','Predicted Avg Sale (₱)':'₱{:,.2f}','Predicted Sales (₱)':'₱{:,.2f}'}),use_container_width=True,height=560)
                
                # --- FORECAST ACCURACY EVALUATOR ---
                with st.expander("🎯 Forecast Accuracy Evaluator"):
                    hist_df = st.session_state.historical_df.copy()
                    fcst_df = st.session_state.forecast_df.copy()

                    if not hist_df.empty and not fcst_df.empty and 'date' in hist_df.columns:
                        # Prepare dataframes for merging
                        hist_df['date'] = pd.to_datetime(hist_df['date']).dt.normalize()
                        fcst_df['ds'] = pd.to_datetime(fcst_df['ds']).dt.normalize()

                        # Find the intersection of dates where forecast was made and actuals are available
                        eval_df = pd.merge(
                            hist_df[['date', 'sales', 'customers', 'add_on_sales']],
                            fcst_df[['ds', 'forecast_sales', 'forecast_customers', 'forecast_atv']],
                            left_on='date',
                            right_on='ds'
                        )

                        if not eval_df.empty:
                            # --- Calculate Actuals based on user request ---
                            eval_df['add_on_sales'] = pd.to_numeric(eval_df['add_on_sales'], errors='coerce').fillna(0)
                            eval_df['actual_total_sales'] = eval_df['sales'] + eval_df['add_on_sales']
                            
                            # Calculate actual ATV using total sales, handle division by zero
                            eval_df['actual_atv'] = np.divide(
                                eval_df['actual_total_sales'],
                                eval_df['customers'],
                                out=np.zeros_like(eval_df['actual_total_sales'], dtype=float),
                                where=eval_df['customers'] != 0
                            )

                            # --- Calculate Error Metrics (MAE) ---
                            sales_mae = mean_absolute_error(eval_df['actual_total_sales'], eval_df['forecast_sales'])
                            customers_mae = mean_absolute_error(eval_df['customers'], eval_df['forecast_customers'])
                            atv_mae = mean_absolute_error(eval_df['actual_atv'], eval_df['forecast_atv'])

                            # --- Display Summary Metrics ---
                            st.markdown("##### Overall Accuracy (Mean Absolute Error)")
                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric(label="Sales Accuracy", value=f"₱{sales_mae:,.2f}", help="Lower is better. The average absolute difference between predicted and actual total sales.")
                            m_col2.metric(label="Customer Accuracy", value=f"{customers_mae:,.1f}", help="Lower is better. The average absolute difference between predicted and actual customers.")
                            m_col3.metric(label="Avg. Sale Accuracy", value=f"₱{atv_mae:,.2f}", help="Lower is better. The average absolute difference between predicted and actual average sale.")
                            st.markdown("---")

                            # --- Display Detailed Breakdown Table ---
                            st.markdown("##### Daily Breakdown Comparison")
                            eval_df['sales_variance'] = eval_df['forecast_sales'] - eval_df['actual_total_sales']
                            eval_df['cust_variance'] = eval_df['forecast_customers'] - eval_df['customers']
                            
                            display_eval_df = eval_df[[
                                'date', 'forecast_sales', 'actual_total_sales', 'sales_variance',
                                'forecast_customers', 'customers', 'cust_variance'
                            ]].copy()

                            # CORRECTED: Renamed columns to remove parentheses
                            display_eval_df.rename(columns={
                                'date': 'Date', 'forecast_sales': 'Predicted Sales', 'actual_total_sales': 'Actual Sales',
                                'sales_variance': 'Sales Variance', 'forecast_customers': 'Predicted Cust.',
                                'customers': 'Actual Cust.', 'cust_variance': 'Customer Variance'
                            }, inplace=True)

                            # CORRECTED: Updated style.format keys and bar subset to match new column names
                            st.dataframe(
                                display_eval_df.set_index('Date').style.format({
                                    'Predicted Sales': '₱{:,.2f}', 
                                    'Actual Sales': '₱{:,.2f}',
                                    'Sales Variance': '₱{:,.2f}', 
                                    'Predicted Cust.': '{:,.0f}',
                                    'Actual Cust.': '{:,.0f}', 
                                    'Customer Variance': '{:,.0f}'
                                }).bar(subset=["Sales Variance", "Customer Variance"], align='mid', color=['#d62728', '#2ca02c']),
                                use_container_width=True
                            )
                        else:
                            st.info("No overlapping past forecast data found to evaluate. Once a forecast is generated and new daily records are added, this section will populate.")
                    else:
                        st.info("Generate a forecast and add historical data to enable accuracy evaluation.")

                st.markdown("#### Forecast Visualization");fig=go.Figure();fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_sales'],mode='lines+markers',name='Sales Forecast',line=dict(color='#ffc72c')));fig.add_trace(go.Scatter(x=future_forecast_df['ds'],y=future_forecast_df['forecast_customers'],mode='lines+markers',name='Customer Forecast',yaxis='y2',line=dict(color='#c8102e')));fig.update_layout(title='15-Day Sales & Customer Forecast',xaxis_title='Date',yaxis=dict(title='Predicted Sales (₱)',color='#ffc72c'),yaxis2=dict(title='Predicted Customers',overlaying='y',side='right',color='#c8102e'),legend=dict(x=0.01,y=0.99,orientation='h'),height=500,margin=dict(l=40,r=40,t=60,b=40),paper_bgcolor='#2a2a2a',plot_bgcolor='#2a2a2a',font_color='white');st.plotly_chart(fig,use_container_width=True)
                with st.expander("🔬 View Full Forecast vs. Historical Data"):
                    st.info("This view shows how the component models performed against past data.");d_t1,d_t2=st.tabs(["Customer Analysis","Avg. Transaction Analysis"]);hist_atv=calculate_atv(st.session_state.historical_df.copy())
                    with d_t1:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_customers':'yhat'}),st.session_state.metrics.get('customers',{}),'customers'),use_container_width=True)
                    with d_t2:st.plotly_chart(plot_full_comparison_chart(hist_atv,st.session_state.forecast_df.rename(columns={'forecast_atv':'yhat'}),st.session_state.metrics.get('atv',{}),'atv'),use_container_width=True)
            else:st.info("Click the 'Generate Forecast' button to begin.")
