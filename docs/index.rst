===============================
Welcome to Amazon Accessible RL
===============================

Amazon Accessible RL (A2RL) provides everything a data scientist needs to develop a solution to a
`sequential decision making problem <https://en.wikipedia.org/wiki/Sequential_decision_making>`_
working on time series data.

You can install the stable version A2RL with ``pip``, preferrably to a virtual environment.

.. code-block:: bash

    pip install a2rl

We aim to bring you a low-code package to apply `offline RL
<https://sites.google.com/view/offlinerltutorial-neurips2020/home>`_, starting from problem
formulation, :doc:`initial data analysis <auto-notebooks/data_properties>` to see if a solution is
possible, train a :doc:`simulator <auto-notebooks/simulator>` (aka *digital twin*) based on the
data, and providing :doc:`recommended actions <auto-notebooks/planner_byo_example>`. At the core of
A2RL is a state-of-the-art generative transformer -- the same technology behind `GATO
<https://www.deepmind.com/publications/a-generalist-agent>`_, `trajectory transformer
<https://trajectory-transformer.github.io/>`_ and `decision transformer
<https://arxiv.org/abs/2106.01345>`_.

You should start by formulating your problem into *states*, *actions*, and *rewards* (see
real-world examples below), then :doc:`prepare a dataset <auto-notebooks/create_dataset>` that reflects
the formulation.

.. collapse:: Manufacturing: Building HVAC system

    .. list-table::
        :widths: 50 50
        :header-rows: 1

        * - Description
          - Formulation
        * - Consider a manufacturing facility with an `HVAC
            <https://en.wikipedia.org/wiki/Heating,_ventilation,_and_air_conditioning>`_ system that
            manages temperature of various equipment, with multiple water `chillers
            <https://en.wikipedia.org/wiki/Chiller>`_ and roof-top `cooling towers
            <https://en.wikipedia.org/wiki/Cooling_tower>`_.

            The building management wants to maximize the chillers efficiency, `COP
            <https://en.wikipedia.org/wiki/Coefficient_of_performance>`_ (Coefficient of
            Performance, in KW/RT), by dynamically `staging the chillers
            <http://www.ibpsa.org/proceedings/BS2015/p2166.pdf>`_ (i.e., deciding which chillers to
            run) to meet dynamic heating load on various weather condition.
          - To frame the problem into "given *states*, decide *actions* that optimize *rewards*",
            the operator would just need to pull these information from their historical data which
            their cooling system collected at an hourly interval.

            .. code-block:: yaml

                states:
                - ECWT (Entering Condenser Water Temperature)
                - evaporator thermal power output (KW or RT)

                actions:
                - identifiers of running chillers

                rewards:
                - power consumption (KW) to minimize

.. collapse:: Manufacturing: Industrial-scale hot stamping

    `Hot stamping <https://en.wikipedia.org/wiki/Hot_stamping>`_ equipment needs to maintain
    constant temperature on dies. Blanks are pre-heated and then stamped and cooled between the
    die. The change in temperature is critical for the part to be annealed to the right
    hardness. This brings opportunities to increase the efficiency of production line, in
    particular to prevent over-heating in the hotstamping line using just minimum amount of
    energy.

    While the exact historical data varies across the systems, in-principle they consist of
    historical staging activities, thermal load, flow rate, temperature, and power consumption.

    .. list-table::
        :widths: 50 50
        :header-rows: 1

        * - Description
          - Formulation
        * - Water `chiller <https://en.wikipedia.org/wiki/Chiller>`_ circuit `rejects heat
            <https://en.wikipedia.org/wiki/Heat_transfer>`_ generated in the die to the atmosphere.
          - Stage chillers (*actions*) based on heat load and ambient temperature (*states*) to
            maximize efficiency (rewards). Since the efficency (`COP
            <https://en.wikipedia.org/wiki/Coefficient_of_performance>`_) is inversely proposional
            to power consumption (KW) which is directly measureable, we can recast the objective as
            minimizing power consumption (*rewards*).
        * - `Furnaces
            <https://www.process-heating.com/articles/93144-video-highlights-hot-stamping-furnace>`_
            heat metal blanks to the right temperature for stamping.
          - An optimization problem statement could be to stage furnaces (*actions*) depending on
            utilization and thermal mass of metal blanks (*states*) to minimize the energy
            consumption (*rewards*).
        * - `Compressor <https://en.wikipedia.org/wiki/Compressor>`_ supplies `compressed air
            <https://en.wikipedia.org/wiki/Compressed_air>`_ for many industrial processes.
          - An optimization problem statement could be to staging compressors (*actions*) based on
            air demand (*states*) to minimize energy consumption (*rewards*).

.. collapse:: Utility: Smart building HVAC economizer

    .. list-table::
        :widths: 50 50
        :header-rows: 1

        * - Description
          - Formulation
        * - HVAC `economizer <https://en.wikipedia.org/wiki/Economizer>`_ optimization is a common
            use case in smart building management. Building operator uses `rooftop unit (RTU)
            <https://en.wikipedia.org/wiki/Air_handler>`_ economizer setpoint to control the amount
            of outside air intake based on weather condition, to maintain an acceptable level of
            comfort in a zone while minimizing power consumption. The target temperature and
            humidity of the zone depends on the actual activity.

            Power is consumed by mechanical cooling to cool the mixture of outside air and returned
            air to the zone's target temperature and humidity. Power consumption depends on
            occupancy rate, outdoor temperature and humidity, returned air temperature and humidity,
            target zone's temperature and humidity, date and time of day, and the activities in the
            zone.
          - To frame this into sequential decision problem, we can define the *states*, *actions*,
            *rewards* as follows:

            .. code-block:: yaml

                states:
                  - occupancy rate
                  - outdoor temperature
                  - outdoor humidity
                  - returned air temperature
                  - returned air humidity

                actions:
                  - RTU economiser setpoints

                rewards:
                  - power consumption

.. collapse:: Utility: Energy arbitrage

    .. list-table::
        :widths: 50 50
        :header-rows: 1

        * - Description
          - Formulation
        * - Energy storage system (ESS) can benefit the grid in many ways such as to balance and
            maintain the grid, or to store electricity for later use during peak demand, outage or
            emergency period. Energy storage has also created new opportunity for energy storage
            owner to generate profit via arbitrage, the difference between revenue received from
            energy sale (discharge) and the charging cost.

            Profit generation from arbitrage depends on energy price uncertainty in real time
            market, and also utility energy battery storage level, cost of energy generation.
            Utility operator need to decide whether to sell, buy or hold on to available energy at
            the right time in order to maximize profit.
          - To frame this into sequential decision problem, we can set the *states*, *actions*,
            *rewards* as follows:

            .. code-block:: yaml

                states:
                  - electric price
                  - battery storage level
                  - battery capacity
                  - charging/discharging efficiency
                  - wear and tear cost
                  - energy demand

                actions:
                  - buy, sell or hold

                rewards:
                  - profit (= energy_sales - charging cost)

            See also a `solution that predates Amazon Whatif
            <https://github.com/aws-samples/sagemaker-rl-energy-storage-system>`_ for this exact
            problem.

.. collapse:: Utility: District heating

    .. list-table::
        :widths: 50 50
        :header-rows: 1

        * - Description
          - Formulation
        * - In a district heating network, the heat generated by producing plant is distributed to
            consumer via heated supply water for floor heating purpose. After the heat has been
            transferred to floor heating, the cold return water is circulated back to the district
            heating plant. The water circulates in a closed pipeline. In this use case, we are
            looking at controlling room temperature using underfloor heating by adjusting the amount
            of heated water flowing into the underfloor pipe. The amount of heated water depends on
            external condition such as outside air temperature and humidity, date and time of day.
          - To frame this into sequential decision problem, we can set the *states*, *actions*,
            *rewards* as follows:

            .. code-block:: yaml

                states:
                  - outside air temperature and humidity
                  - room temperature
                  - supply water temperature
                  - occupancy rate

                actions:
                  - return water temperature setpoint

                rewards:
                  - temperature differences

            where ``temperature differences`` is the discrepancy between actual and target room
            temperature.

| Once the dataset is ready, the end-to-end workflow is as concise as follows:

.. code-block:: python

    import a2rl as wi
    from a2rl.utils import plot_information

    # Load a sample dataset which contains historical states, actions, and rewards.
    wi_df = wi.read_csv_dataset(wi.sample_dataset_path("chiller")).trim().add_value()
    wi_df = wi_df.iloc[:1000]  # Reduce data size for demo purpose

    # Checks and analysis
    plot_information(wi_df)

    # Train a simulator
    tokenizer = wi.AutoTokenizer(wi_df, block_size_row=2)
    builder = wi.GPTBuilder(tokenizer, model_dir="my-model", )
    model = builder.fit()
    simulator = wi.Simulator(tokenizer, model, max_steps=100, reset_coldstart=2)

    # Get recommended actions given an input context (s,a,r,v,...s).
    # Context must end with states, and its members must be tokenized.
    custom_context = simulator.tokenizer.df_tokenized.sequence[:7]
    recommendation_df = simulator.sample(custom_context, 3)

    # Show recommendations (i.e., trajectory)
    recommendation_df

Contents
========

.. toctree::
    :hidden:

    Overview <self>

.. toctree::
    :maxdepth: 1

    start_here
    example
    changelog
    reference
    developer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
